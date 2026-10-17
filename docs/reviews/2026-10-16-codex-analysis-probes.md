# Analysis probe provenance and separation

The CLI analysis wrapper now binds diagnostics to a replayable validation probe plan: validation manifest SHA256, context, initial sampler state, source-document ranges for every batch, and pre-call batch hashes. The plan has a canonical JSON SHA256. The final sampler state remains recorded. The wrapper replaces any provenance supplied by the opaque diagnostic callback with this authoritative record.

Before invoking diagnostics, the wrapper audits selected input documents for exact original or processed whole-document overlap with training, calibration and final-test metadata, plus duplicate selected validation content. Lookahead-only documents are excluded because they are not diagnostic inputs. Final-test token streams are never sampled. This detects exact whole-document collisions, not partial duplication or general statistical dependence.

After diagnostics, input, document-ID and target tensor hashes must match their pre-call values, alongside the existing model/buffer/mode/hook checks. Failure leaves the ledger failed and publishes no completed invocation result. Diagnostic-owned partial files may remain if the callback wrote files before failing; they are not successful wrapper results.

Independent tests replay batches across epochs with batch size two, reject exact and processed content overlap and duplicated validation content, check final-test metadata separation and lookahead exclusion, overwrite forged callback provenance, and detect mutation of each supplied tensor type.

A separate read-only audit found repeated windows from the same document are treated as independent documents in Claude-owned ablation statistics. This changes the reported mean as well as its standard error. Reproduction and numeric evidence: `runs/codex_analysis_audit_followup/`. The request is recorded in COORDINATION.md; this wrapper change does not repair that math or the earlier missing residual/readout scaling factors.

Validation: 13 focused tests pass; full isolated CPU regression passes 583 tests in 158.61 seconds with three warnings. Claude kernel/parallel test files excluded; no GPU validation or commits.
