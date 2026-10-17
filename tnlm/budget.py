"""Locked project and stage spending admission and accounting."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path


STAGE_BUDGETS = {'A': 150.0, 'B': 175.0, 'C': 175.0, 'D': 350.0, 'Reserve': 150.0}



class SpendLedger:
    """Locked reservations prevent parallel jobs from allocating the same dollars.

    Crashed jobs retain their reservation until manually reconciled. Step-boundary
    limits cannot preempt a CUDA operation; the measured bill always records actual
    elapsed time, including any final step/checkpoint overrun.
    """
    def __init__(self, path, total_budget=1000.0, reserve=150.0, stage_budgets=None):
        self.path = Path(path)
        self.total, self.reserve = float(total_budget), float(reserve)
        if not math.isfinite(self.total) or not math.isfinite(self.reserve) or not 0 <= self.reserve <= self.total:
            raise ValueError("invalid project budget/reserve")
        if stage_budgets is not None and (not isinstance(stage_budgets, dict) or set(stage_budgets) - set(STAGE_BUDGETS)):
            raise ValueError('unknown stage budgets')
        self.stage_budgets = {key: float(value) for key, value in {**STAGE_BUDGETS, **(stage_budgets or {})}.items()}
        if any(not math.isfinite(value) or value < 0 for value in self.stage_budgets.values()):
            raise ValueError('stage budgets must be finite and nonnegative')
        self.stage_budgets['Reserve'] = min(self.reserve, self.stage_budgets['Reserve'])

    @contextlib.contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "runs": {}}
            yield data
            tmp = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
            with open(tmp, 'w') as out:
                json.dump(data, out, indent=2, allow_nan=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, self.path)

    def begin(self, run_id, hourly_rate, max_seconds, run_budget=None, *, parent_checkpoint=None, cost_margin_fraction=0.0, stage='A'):
        if stage not in self.stage_budgets:
            raise ValueError(f'unknown budget stage: {stage}')
        rate, seconds = float(hourly_rate), float(max_seconds)
        if not math.isfinite(rate) or not math.isfinite(seconds) or rate < 0 or seconds <= 0:
            raise ValueError("finite nonnegative rate and positive runtime required")
        margin = float(cost_margin_fraction)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError('cost margin must be finite and nonnegative')
        reservation = rate * seconds / 3600 * (1 + margin)
        if run_budget is not None and (not math.isfinite(run_budget) or reservation > run_budget):
            raise ValueError("runtime reservation exceeds run budget")
        with self.locked() as data:
            if run_id in data['runs']:
                raise ValueError(f"duplicate run ID: {run_id}")
            allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0) for r in data['runs'].values())
            normal_allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0)
                for r in data['runs'].values() if r.get('stage', 'A') != 'Reserve')
            stage_allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0)
                for r in data['runs'].values() if r.get('stage', 'A') == stage)
            if allocated + reservation > self.total or (stage != 'Reserve' and normal_allocated + reservation > self.total - self.reserve):
                raise ValueError("project budget exhausted (including reserve and active runs)")
            if stage_allocated + reservation > self.stage_budgets[stage]:
                raise ValueError(f'stage {stage} budget exhausted')
            data['runs'][run_id] = {"status": "running", "rate_usd_per_hour": rate,
                                    "reserved_usd": reservation, "spent_usd": 0.0,
                                    "parent_checkpoint": str(Path(parent_checkpoint).resolve()) if parent_checkpoint else None,
                                    "cost_margin_fraction": margin, 'stage': stage, 'kind': 'runtime'}
        return reservation

    def finish(self, run_id, elapsed, status):
        with self.locked() as data:
            row = data['runs'][run_id]
            if row['status'] != 'running':
                raise ValueError('only an active runtime reservation can be finalized')
            row.update(status=status, elapsed_seconds=elapsed,
                       spent_usd=elapsed * row['rate_usd_per_hour'] / 3600, reserved_usd=0.0)


    def record_charge(self, charge_id, usd, *, stage='Reserve', description):
        """Record an observed API/storage/provider charge, even if it exceeds a cap.

        Paid charges cannot be undone by rejecting bookkeeping. Actual overruns
        remain visible and prevent further admissions. No external API is called.
        """
        usd = float(usd)
        if stage not in self.stage_budgets or not math.isfinite(usd) or usd < 0 or not description:
            raise ValueError('known stage, finite nonnegative charge and description required')
        with self.locked() as data:
            if charge_id in data['runs']:
                raise ValueError(f'duplicate run/charge ID: {charge_id}')
            row = {'kind': 'external', 'status': 'charged', 'stage': stage,
                   'spent_usd': usd, 'reserved_usd': 0.0, 'description': description}
            data['runs'][charge_id] = row
            allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0) for r in data['runs'].values())
            stage_allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0)
                for r in data['runs'].values() if r.get('stage', 'A') == stage)
            row['exceeds_project_budget'] = allocated > self.total
            row['exceeds_stage_budget'] = stage_allocated > self.stage_budgets[stage]
        return row


    @staticmethod
    def _row_digest(row):
        return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _reconciliation_evidence(run_id, evidence_paths):
        evidence, observed = [], 0.0
        for filename in evidence_paths:
            path = Path(filename).resolve()
            raw = path.read_bytes()
            # Explicitly supplied JSON or JSONL only; incomplete lines fail closed.
            try:
                decoded = json.loads(raw)
                records = decoded if isinstance(decoded, list) else [decoded]
            except json.JSONDecodeError:
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError('metrics evidence must contain objects')
                identity = record.get('run_id', record.get('run_dir'))
                if identity is not None and str(Path(identity).resolve()) != str(Path(run_id).resolve()):
                    raise ValueError('metrics evidence run identity mismatch')
                value = record.get('run_elapsed_seconds')
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid observed elapsed seconds')
                    observed = max(observed, float(value))
            evidence.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                             'bytes': len(raw), 'identity_verified': all(
                                 r.get('run_id', r.get('run_dir')) is not None for r in records)})
        return evidence, observed

    def inspect_reconciliation(self, run_id, *, evidence_paths=()):
        """Inspect supplied evidence without creating or rewriting a ledger.

        Metrics are elapsed-time lower bounds, never proof of job termination.
        Evidence lacking run identity is explicitly marked unverified.
        """
        with open(str(self.path) + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            data = json.loads(self.path.read_text())
            row = copy.deepcopy(data['runs'][run_id])
        if row.get('kind', 'runtime') != 'runtime' or row['status'] not in ('running', 'reconciled_unknown_tail'):
            raise ValueError('only unresolved runtime entries can be reconciled')
        evidence, elapsed = self._reconciliation_evidence(run_id, evidence_paths)
        return {'run_id': run_id, 'row': row, 'row_sha256': self._row_digest(row),
                'evidence': evidence, 'observed_elapsed_seconds': elapsed,
                'observed_cost_lower_bound_usd': max(row.get('spent_usd', 0.0), elapsed * row['rate_usd_per_hour'] / 3600),
                'termination_verified': False, 'tail_cost_known': False}

    def reconcile_terminal(self, run_id, *, expected_row_sha256, operator, reason,
                           terminal_attestation, observed_elapsed_seconds=None,
                           total_elapsed_seconds=None, total_cost_usd=None, evidence_paths=()):
        """Explicit operator bookkeeping, not process detection or stage adoption.

        terminal_attestation must be exactly {'run_id': run_id, 'all_workers_stopped': True}.
        A lower bound retains the original allocation; an explicit total releases
        reservation. Unverified evidence requires operator attribution. No process
        or provider API is contacted, and completed training is never inferred.
        """
        if terminal_attestation != {'run_id': run_id, 'all_workers_stopped': True} or terminal_attestation.get('all_workers_stopped') is not True:
            raise ValueError('explicit identified-run terminal attestation required')
        if not isinstance(operator, str) or not operator.strip() or not isinstance(reason, str) or not reason.strip():
            raise ValueError('operator and reason required')
        for value in (observed_elapsed_seconds, total_elapsed_seconds, total_cost_usd):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError('finite nonnegative reconciliation amounts required')
        if total_elapsed_seconds is not None and total_cost_usd is not None:
            raise ValueError('supply one authoritative total')
        evidence, elapsed = self._reconciliation_evidence(run_id, evidence_paths)
        elapsed = max(elapsed, observed_elapsed_seconds or 0.0)
        with self.locked() as data:
            row = data['runs'][run_id]
            if self._row_digest(row) != expected_row_sha256:
                raise ValueError('ledger row changed since inspection')
            if row.get('kind', 'runtime') != 'runtime' or row['status'] not in ('running', 'reconciled_unknown_tail'):
                raise ValueError('only unresolved runtime entries can be reconciled')
            old = copy.deepcopy(row)
            elapsed = max(elapsed, *(a.get('observed_elapsed_seconds', 0.0) for a in row.get('reconciliations', []))) if row.get('reconciliations') else elapsed
            lower = max(row.get('spent_usd', 0.0), elapsed * row['rate_usd_per_hour'] / 3600)
            total = total_cost_usd
            if total_elapsed_seconds is not None:
                if total_elapsed_seconds < elapsed:
                    raise ValueError('total elapsed below observed lower bound')
                total = total_elapsed_seconds * row['rate_usd_per_hour'] / 3600
            if total is not None and total < lower:
                raise ValueError('total cost below observed lower bound')
            if not math.isfinite(lower) or (total is not None and not math.isfinite(total)):
                raise ValueError('reconciliation cost overflow')
            known = total is not None
            allocation = row.get('spent_usd', 0.0) + row.get('reserved_usd', 0.0)
            row.update(status='reconciled_terminal' if known else 'reconciled_unknown_tail',
                       spent_usd=total if known else lower,
                       reserved_usd=0.0 if known else max(0.0, allocation - lower))
            audit = {'operator': operator, 'reason': reason,
                     'terminal_attestation': copy.deepcopy(terminal_attestation),
                     'old_row': old, 'old_row_sha256': expected_row_sha256,
                     'observed_elapsed_seconds': elapsed, 'total_elapsed_seconds': total_elapsed_seconds,
                     'total_cost_usd': total_cost_usd, 'tail_cost_known': known, 'evidence': evidence}
            row.setdefault('reconciliations', []).append(audit)
            result = copy.deepcopy(row)
            result_digest = self._row_digest(row)
        return {'run_id': run_id, 'row': result, 'row_sha256': result_digest,
                'tail_cost_known': known, 'training_completion_verified': False}
