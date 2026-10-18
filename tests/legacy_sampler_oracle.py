"""Frozen pre-index sampler oracle, captured 2026-10-18 from tnlm/data.py.

Kept independent of the compact index implementation to detect ordering, state,
packing, fork, rank, or metadata regressions. No git executable is needed.
"""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from tnlm.data import _sha, manifest_hash

SPLIT_ORDER = ('train', 'val', 'test', 'calibration')
PACKING_VERSION = 'document-shuffle-row-overlap-incremental-v2'

class PackedSampler:
    """Deterministic epoch permutations, row packing, and resumable next-batch order."""
    def __init__(self, manifest, split='train', batch_size=1, context=512, seed=0):
        self.path = Path(manifest)
        if self.path.is_dir():
            self.path /= 'manifest.json'
        self.manifest_hash = manifest_hash(self.path)
        self.manifest = json.loads(self.path.read_text())
        root = self.path.parent
        for name, expected in self.manifest['artifacts'].items():
            if Path(name).name != name or _sha(root / name) != expected:
                raise ValueError(f'artifact checksum mismatch: {name}')
        if split not in self.manifest['tokens'] or batch_size <= 0 or context <= 0:
            raise ValueError('invalid split or batch shape')
        self.split, self.batch_size, self.context, self.seed = split, batch_size, context, seed
        required = {'documents.jsonl', *(f'{name}.tokens.bin' for name in SPLIT_ORDER)}
        if not required.issubset(self.manifest['artifacts']):
            raise ValueError('dataset manifest lacks required metadata/token artifacts')
        if self.manifest.get('format_version') != 1:
            raise ValueError('unsupported dataset format')
        counts = self.manifest['tokens']
        if set(counts) != set(SPLIT_ORDER) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('invalid dataset token counts')
        for name, count in counts.items():
            if (root / f'{name}.tokens.bin').stat().st_size != count * 4:
                raise ValueError(f'token payload size mismatch: {name}')
        self.docs = []
        seen_ids, seen_numeric_ids = set(), set()
        covered = dict.fromkeys(SPLIT_ORDER, 0)
        document_count = 0
        with (root / 'documents.jsonl').open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    doc = json.loads(line)
                    name = doc['split']
                    doc_id, numeric_id = doc['id'], doc['numeric_id']
                    start, end = doc['token_start'], doc['token_end']
                    if name not in covered or not isinstance(doc_id, str) or doc_id in seen_ids:
                        raise ValueError('invalid split or duplicate document ID')
                    if type(numeric_id) is not int or numeric_id < 0 or numeric_id in seen_numeric_ids:
                        raise ValueError('invalid or duplicate numeric document ID')
                    if type(start) is not int or type(end) is not int or start != covered[name] or not start < end <= counts[name]:
                        raise ValueError('document token ranges must cover each split contiguously')
                    for field in ('content_sha256', 'processed_sha256'):
                        digest = doc[field]
                        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest.lower()):
                            raise ValueError(f'invalid {field}')
                    seen_ids.add(doc_id)
                    seen_numeric_ids.add(numeric_id)
                    covered[name] = end
                    document_count += 1
                    if name == split:
                        self.docs.append(doc)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'invalid document metadata at line {line_number}: {error}') from error
        if covered != counts or document_count != self.manifest.get('documents'):
            raise ValueError('document metadata does not cover declared dataset')
        if not self.docs:
            raise ValueError(f'empty split: {split}')
        self.tokens = np.memmap(root / f'{split}.tokens.bin', dtype='<i4', mode='r')
        self.packing_version = PACKING_VERSION
        self.epoch = self.cursor = 0
        self._order()
        self._batch_mappings = []
        self._context_transition = None

    def _permutation(self, epoch):
        order = list(range(len(self.docs)))
        random.Random(f'{self.seed}:{epoch}').shuffle(order)
        return order

    def _count_lengths(self, lengths):
        # O(documents), including context=1 and very long individual documents.
        rows = fill = 0
        width = self.context + 1
        for remaining in lengths:
            capacity = width - fill
            if remaining < capacity:
                fill += remaining
                continue
            rows += 1
            remaining -= capacity
            fill = 0
            if remaining:
                remaining += 1
                complete = (remaining - 1) // self.context
                rows += complete
                fill = remaining - complete * self.context
                if complete and fill == 1:
                    fill = 0
        return rows, fill

    def _order(self):
        # A one-row replay mask belongs only to its source epoch. A terminal
        # singleton can place the fork at epoch end before the mask is consumed.
        self._context_transition = None
        self.order = self._permutation(self.epoch)
        rows, fill = self._count_lengths(self.docs[i]['token_end'] - self.docs[i]['token_start'] for i in self.order)
        self.epoch_row_count = rows + bool(fill)
        self.document_cursor = self.token_cursor = 0

    def _seek_row(self, order, cursor):
        """O(documents) canonical row-boundary seek; no token/row materialization."""
        d = t = fill = 0
        remaining_rows = cursor
        width = self.context + 1
        while remaining_rows:
            if d == len(order):
                if fill and remaining_rows == 1:
                    return d, 0
                raise ValueError('invalid canonical row seek')
            doc = self.docs[order[d]]
            length = doc['token_end'] - doc['token_start']
            remaining = length - t
            if fill == 0 and remaining >= width:
                count = min(remaining_rows, (remaining - 1) // self.context)
                t += count * self.context
                remaining_rows -= count
                if t == length - 1:
                    d, t = d + 1, 0
                continue
            capacity = width - fill
            if remaining < capacity:
                fill += remaining
                d, t = d + 1, 0
            else:
                if remaining == capacity:
                    d, t = d + 1, 0
                else:
                    t += capacity - 1
                remaining_rows -= 1
                fill = 0
        return d, t

    def _fork_position(self, source):
        epoch, d, t = source['epoch'], source['document_cursor'], source['token_cursor']
        order = self._permutation(epoch)
        if d == len(order):
            epoch, d, t = epoch + 1, 0, 0
            order = self._permutation(epoch)
        # Number of complete canonical new-context rows up to the scored token.
        lengths = [self.docs[i]['token_end'] - self.docs[i]['token_start'] for i in order[:d]]
        cursor, _ = self._count_lengths((*lengths, t + 1))
        document_cursor, token_cursor = self._seek_row(order, cursor)
        return epoch, cursor, document_cursor, token_cursor, d, t

    def fork_context(self, context):
        """Return a new packing stream, preserving consumed-stream target watermark.

        Earlier canonical inputs may replay as prefix but their targets are masked
        once. This certifies consumed sampler rows, not optimizer quota selection:
        callers may previously have discarded overflow targets. Read-only corpus
        storage is shared; original cursor and last-batch mappings stay unchanged.
        """
        import copy
        if type(context) is not int or context <= 0:
            raise ValueError('context must be a positive integer')
        original = PackedSampler.state_dict(self)
        check = copy.copy(self)
        PackedSampler.load_state_dict(check, original)
        if context == self.context:
            return check
        transition = original.get('context_transition')
        source = copy.deepcopy(transition['source_state'] if transition else original)
        fork = copy.copy(self)
        fork.context = context
        epoch, cursor, d, t, _, _ = fork._fork_position(source)
        target = {**source, 'context': context, 'epoch': epoch, 'cursor': cursor,
                  'document_cursor': d, 'token_cursor': t,
                  'context_transition': {'version': 1, 'source_context': source['context'], 'source_state': source}}
        PackedSampler.load_state_dict(fork, target)
        return fork

    def _next_row(self):
        """Build one row only, retaining no prior/future packed rows."""
        row, fill = [], 0
        width = self.context + 1
        while self.document_cursor < len(self.order) and fill < width:
            i = self.order[self.document_cursor]
            doc = self.docs[i]
            length = doc['token_end'] - doc['token_start']
            count = min(width - fill, length - self.token_cursor)
            row.append((i, doc['token_start'] + self.token_cursor, count))
            fill += count
            self.token_cursor += count
            if self.token_cursor == length:
                self.document_cursor += 1
                self.token_cursor = 0
            elif fill == width:
                self.token_cursor -= 1
        return row

    def next_batch(self):
        idx = torch.full((self.batch_size, self.context), 50256, dtype=torch.long)
        ids = torch.full_like(idx, -1)
        targets = torch.full_like(idx, -100)
        self._batch_mappings = []
        for b in range(self.batch_size):
            if self.cursor >= self.epoch_row_count:
                self.epoch += 1
                self.cursor = 0
                self._order()
            transition = self._context_transition
            row = self._next_row()
            self._context_transition = None
            self.cursor += 1
            tok, docids, mappings = [], [], []
            for i, start, count in row:
                doc = self.docs[i]
                packed_start = len(tok)
                mappings.append({'document_id': doc['id'], 'numeric_id': doc['numeric_id'],
                                 'content_sha256': doc['content_sha256'],
                                 'processed_sha256': doc['processed_sha256'],
                                 'source_offset': doc['source_offset'],
                                 'synthetic_eos_token_offset': doc['synthetic_eos_token_offset'],
                                 'processed_document_token_count': doc['processed_token_end'],
                                 'packed_token_start': packed_start,
                                 'packed_token_end': packed_start + count,
                                 'source_token_start': start - doc['token_start'],
                                 'source_token_end': start - doc['token_start'] + count})
                tok.extend(self.tokens[start:start + count].tolist())
                docids.extend([self.docs[i]['numeric_id']] * count)
            self._batch_mappings.append({'epoch': self.epoch, 'row_cursor': self.cursor - 1,
                                         'input_length': min(self.context, len(tok)),
                                         'ranges': mappings})
            n = min(self.context, len(tok))
            token_row = torch.tensor(tok, dtype=torch.long)
            document_row = torch.tensor(docids, dtype=torch.long)
            idx[b, :n] = token_row[:n]
            ids[b, :n] = document_row[:n]
            pairs = min(self.context, len(tok) - 1)
            if pairs > 0:
                same_document = document_row[:pairs] == document_row[1:pairs + 1]
                targets[b, :pairs] = torch.where(same_document, token_row[1:pairs + 1], -100)
            if transition is not None:
                source = transition['source_state']
                _, _, _, _, frontier_doc, frontier_token = self._fork_position(source)
                positions = {self.docs[i]['numeric_id']: position for position, i in enumerate(self.order)}
                exposure = {'prefix_tokens_at_or_before_watermark': 0, 'masked_prediction_targets': 0}
                for mapping in mappings:
                    position = positions[mapping['numeric_id']]
                    for packed in range(mapping['packed_token_start'], min(n, mapping['packed_token_end'])):
                        offset = mapping['source_token_start'] + packed - mapping['packed_token_start']
                        if position < frontier_doc or (position == frontier_doc and offset <= frontier_token):
                            exposure['prefix_tokens_at_or_before_watermark'] += 1
                    for packed in range(mapping['packed_token_start'], min(self.context, mapping['packed_token_end'] - 1)):
                        target_offset = mapping['source_token_start'] + packed - mapping['packed_token_start'] + 1
                        if position < frontier_doc or (position == frontier_doc and target_offset <= frontier_token):
                            exposure['masked_prediction_targets'] += int(targets[b, packed] != -100)
                            targets[b, packed] = -100
                self._batch_mappings[-1]['context_fork_exposure'] = exposure
            # Unique negative padding IDs prevent padding from sharing document scope.
            if n < self.context:
                ids[b, n:] = -torch.arange(1, self.context - n + 1)
        return idx, ids, targets

    def batch_mappings(self):
        """Copy last batch lineage; ranges include the target-lookahead slot T.

        Each range maps packed positions [start,end) to source document token
        offsets. Positions >= input_length are lookahead or padding, not inputs.
        Calling this method never advances the sampler.
        """
        import copy
        return copy.deepcopy(self._batch_mappings)

    def batch_context_exposure(self):
        """Local last-batch context-transition counts, excluding padding/lookahead."""
        keys = ('prefix_tokens_at_or_before_watermark', 'masked_prediction_targets')
        return {key: sum(row.get('context_fork_exposure', {}).get(key, 0)
                         for row in self._batch_mappings) for key in keys}

    __next__ = next_batch

    def __iter__(self):
        return self

    def state_dict(self):
        import copy
        state = {k: getattr(self, k) for k in ('packing_version', 'manifest_hash', 'split', 'batch_size', 'context', 'seed', 'epoch', 'cursor', 'document_cursor', 'token_cursor')}
        if self._context_transition is not None:
            state['context_transition'] = copy.deepcopy(self._context_transition)
        return state

    def load_state_dict(self, state):
        for k in ('packing_version', 'manifest_hash', 'split', 'batch_size', 'context', 'seed'):
            if state.get(k) != getattr(self, k):
                raise ValueError(f'sampler state mismatch: {k}')
        epoch, cursor = state['epoch'], state['cursor']
        document_cursor, token_cursor = state['document_cursor'], state['token_cursor']
        if any(type(value) is not int or value < 0 for value in (epoch, cursor, document_cursor, token_cursor)):
            raise ValueError('invalid sampler cursor')
        # Build/validate in locals: a rejected restore must leave the live stream
        # unchanged, including cached last-batch mappings.
        order = self._permutation(epoch)
        rows, fill = self._count_lengths(self.docs[i]['token_end'] - self.docs[i]['token_start'] for i in order)
        epoch_row_count = rows + bool(fill)
        if cursor > epoch_row_count or document_cursor > len(order):
            raise ValueError('invalid sampler cursor')
        if document_cursor == len(order):
            if token_cursor != 0 or cursor != epoch_row_count:
                raise ValueError('inconsistent completed epoch cursor')
        else:
            doc = self.docs[order[document_cursor]]
            length = doc['token_end'] - doc['token_start']
            if token_cursor >= length or cursor == epoch_row_count:
                raise ValueError('invalid document token cursor')
            prefix_lengths = [self.docs[i]['token_end'] - self.docs[i]['token_start'] for i in order[:document_cursor]]
            previous_rows, previous_fill = self._count_lengths(prefix_lengths)
            at_document_start = token_cursor == 0 and previous_fill == 0 and previous_rows == cursor
            if not at_document_start:
                previous_rows, previous_fill = self._count_lengths((*prefix_lengths, token_cursor + 1))
                if token_cursor >= length - 1 or previous_fill != 0 or previous_rows != cursor:
                    raise ValueError('inconsistent row/document/token cursors')
        import copy
        transition = copy.deepcopy(state.get('context_transition'))
        if transition is not None:
            if not isinstance(transition, dict) or set(transition) != {'version', 'source_context', 'source_state'} or type(transition['version']) is not int or transition['version'] != 1:
                raise ValueError('invalid context transition schema')
            source = transition['source_state']
            source_context = transition['source_context']
            if (type(source_context) is not int or source_context <= 0 or not isinstance(source, dict)
                    or type(source.get('context')) is not int or source.get('context') != source_context
                    or set(source) != {'packing_version', 'manifest_hash', 'split', 'batch_size', 'context', 'seed', 'epoch', 'cursor', 'document_cursor', 'token_cursor'}):
                raise ValueError('invalid context transition source')
            checker = copy.copy(self)
            checker.context = source_context
            checker._context_transition = None
            PackedSampler.load_state_dict(checker, source)
            expected = self._fork_position(source)[:4]
            if (epoch, cursor, document_cursor, token_cursor) != expected:
                raise ValueError('context transition cursor mismatch')
        self.epoch, self.order, self.epoch_row_count = epoch, order, epoch_row_count
        self.cursor = cursor
        self.document_cursor, self.token_cursor = document_cursor, token_cursor
        self._context_transition = transition
        self._batch_mappings = []



class RankPackedSampler(PackedSampler):
    """Rank slices of one deterministic global packed-row stream.

    A call consumes ``world_size * batch_size`` rows globally, returning this
    rank's contiguous slice. Every rank must call equally often. Only local rows
    materialize token tensors; other rows advance packing metadata. Epochs may
    repeat within a global batch for tiny corpora, exactly as in PackedSampler.
    This partitions stream positions, not unique document identities.
    """
    def __init__(self, manifest, *, rank, world_size, split='train', batch_size=1,
                 context=512, seed=0):
        if (type(world_size) is not int or world_size < 1 or type(rank) is not int
                or not 0 <= rank < world_size):
            raise ValueError('invalid sampler rank/world_size')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.rank, self.world_size = rank, world_size
        super().__init__(manifest, split=split, batch_size=batch_size, context=context, seed=seed)

    def _skip_rows(self, count):
        for _ in range(count):
            if self.cursor >= self.epoch_row_count:
                self.epoch += 1
                self.cursor = 0
                self._order()
            self._next_row()
            self._context_transition = None
            self.cursor += 1

    def next_batch(self):
        self._skip_rows(self.rank * self.batch_size)
        batch = super().next_batch()
        self._skip_rows((self.world_size - self.rank - 1) * self.batch_size)
        return batch

    __next__ = next_batch

    def state_dict(self):
        return {'rank_packing_version': 1, 'rank': self.rank, 'world_size': self.world_size,
                'stream': super().state_dict()}

    def load_state_dict(self, state):
        for key, expected in (('rank_packing_version', 1), ('rank', self.rank),
                              ('world_size', self.world_size)):
            if type(state.get(key)) is not int or state[key] != expected:
                raise ValueError(f'rank sampler state mismatch: {key}')
        super().load_state_dict(state['stream'])


