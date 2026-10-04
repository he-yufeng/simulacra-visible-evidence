# Visible Evidence for Survey Completion

## R09: whole-block cross-fitted frozen expert mixture

This branch contains the exact next Development candidate, not a new official
score or prize. The original r05 submission959261 remained the recorded
official best at displayed0.58 on5October02:34UTC+8. R09 had not been uploaded;
the live daily quota was1/1. Its locally verified25,278-byte ZIP hashes to
`a5295ed4511cc6a950778cd12b1710cdbf7273b0d93326e4a0c8c2b9e07fc4e8`.

R09 combines the frozen evidence and standard latent-class predictors only
when three-fold visible-row held-out log scores support a fixed half mixture.
The entire PREDICT block is masked before either expert trains; query labels,
identifiers and EXCLUDE fields do not enter model or selector training. If a
target does not pass the fixed internal gain rule, its exact r05 vector is
retained. See [R09_PROTOCOL.md](R09_PROTOCOL.md), the two frozen cores and the
[owned isolation tests](tests/test_participant_r09.py).

The local fixed comparison completed12comparable score cells:11new executions
plus one earlier completed r05 reference reused only after exact source/seed/
phase/data/log checks. All six paired raw gains were positive; five exceeded
the predeclared0.005 threshold, with no material regression. These are only
two public synthetic seeds, not real-microdata gains, significance or awards.
The initial300-second local allocation timed out and was retained. An explicit
[budget revision](R09_RESOURCE_PROTOCOL_R01.md) used600/120/120seconds plus60
reserved overhead within the official900-second shared Development budget,
without changing code, folds, weight, seeds or quality thresholds.
See the bounded [local aggregate](experiments/r09/local_method_summary.json).

The exact ZIP also passed three local final-stage entry contracts covering
442,425legal vectors. That was not a Test score or submission. The independent
Linux job checks the exact same ZIP and all three complete public stand-in
schemas, using the immutable official toolkit and fresh native tests. It
does not rerun the local twelve-cell quality study or use real records. Native
CI state/results remain separate from the local evidence and from H100 or
official ranking proof.

To run the independent native contract, use Linux x86_64/Python3.12, install
`benchmark-requirements.txt`, clone the fixed official commit into `_reference/`,
then run `python -B tools/native_r09.py`. It generates only public synthetic
stand-ins and refuses source/ZIP drift. The public standard35-minute runner
has no paid larger machine, cache, artifact upload, external model API or GPU.
The old stopped r06 workflow is not re-enabled or dispatched by this branch.

Participant-authored, dependency-light categorical probability models for
[SimulacraBench](https://www.codabench.org/competitions/17822/). Only supplied
visible training answers and declared schema relationships are used. No
identifiers, hidden answers, hard skip projections, external models or runtime
network/file access are used by the predictors.

## Actual evidence, not a prize claim

The frozen `participant_r05/main.py` matches the submitted source
`86d0eb535267f338e05e31aae8585198f953202941e1a7cc03fddccfcffc1184`.
Submission **959261** was observed Finished on 4 October 2026: displayed
Development score **0.58** (UNICEF 0.50, World Bank 0.49, UNHCR 0.75).
Earlier submissions displayed 0.45 and 0.54. Development feedback is
Laplace-noised and rounded to 0.01; these are not exact quality differences,
final-Test results, statistical significance, prizes or proof of generalization.
The historical docstring in the frozen r05 file is intentionally retained.

R05 combines multiple GIVEN-feature conditional probability tables using
temperature calibration and nested row-held-out selection, falling back to
simpler single/pair conditional models. R06 is an **unsubmitted experiment**:
a learned soft parent/child gate adjustment, not a hard routing rule.
Its protocol and stopping rule are fixed before the synthetic comparison.
See [R06_PROTOCOL.md](R06_PROTOCOL.md).

## R06 outcome: stopped, not selected

The [fixed Linux comparison](https://github.com/he-yufeng/simulacra-visible-evidence/actions/runs/37173691214)
ended **STOP_FUTILITY** after the first three paired instruments (six scoring
calls). Raw synthetic skill gains were +0.001284 (UNHCR), +0.000921 (UNICEF)
and +0.000462 (World Bank). None reached the predeclared +0.005 useful-gain
threshold. With only three pairs remaining, the required four useful pairs
became unreachable. The remaining seed and final-stage run were not executed,
and r06 was not officially submitted. CI success means the stopping rule ran,
not that the candidate passed promotion. The official displayed best remains
r05 at 0.58.

The first attempt failed because the frozen r05 baseline exceeded a proxy
180-second prediction cap. That failure was retained; before any candidate
score the proxy allocation was explicitly revised to 300 seconds per
instrument (three caps sum to the official Development 900-second budget).
Source, seeds and performance thresholds were unchanged, and the first
generated dataset had to match its original hash. See the complete aggregate
[method receipt](experiments/r06/method_receipt.json). No real microdata was
used or released.

## Reproduce the fixed synthetic comparison

Use Python 3.12 and the versions in `benchmark-requirements.txt`. Check out the
[official MIT toolkit](https://github.com/SituatedEvals/public) at commit
`0d2332d8ae19a8ce171031142bdc97134910e7ec` into `_reference/`.
Run `python -B tools/run_r06.py`. This runs only invented stand-ins from the
public generator, not the confidential competition microdata. It reports raw
synthetic scores separately from reported/noised scores and stops on its
predeclared futility, regression, time or resource gate.

The manually dispatched CI uses standard public GitHub-hosted Linux runners.
It has no scheduled trigger, paid large runner, Actions cache, artifact upload,
model API, GPU purchase, credentials in code, real records or private mail.
Logs contain bounded aggregate metrics/provenance only. CI success means the
protocol executed; a STOP_FUTILITY result does **not** promote the candidate.

## Attribution

Copyright 2026 Yufeng He. Public schema conventions, generator and scoring
tools are by SituatedEvals (Yegor Denisov-Blanch, José Ramón Enríquez and
Andreas Haupt), MIT licensed and independently fetched at the fixed commit.
Their notices are preserved. This repository is a participant project, not an
official toolkit or organizer endorsement.
