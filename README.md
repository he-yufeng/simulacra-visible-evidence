# Visible Evidence for Survey Completion

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
