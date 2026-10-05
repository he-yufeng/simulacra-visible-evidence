# R17: schema-numeric order alongside all R11 categorical channels

Frozen before any R17 questionnaire-quality measurement, 2026-10-06 (UTC+8).

## Evidence and hypothesis

The public official schemas at SituatedEvals/public commit
`0d2332d8ae19a8ce171031142bdc97134910e7ec` contain clearly numerical singleton,
interval and open-ended GIVEN labels: UNHCR age and eight household-count
fields, UNICEF age, World Bank age and two wage fields (13 in total). R11
represents every GIVEN as opaque category IDs. Keeping these channels while
adding order may make contiguous threshold splits more accessible. This is a
new representation hypothesis, not evidence that ordinal information caused
previous errors or a prediction of a leaderboard gain.

No question names, country names, target values or outcome feedback select
features. A strict whole-label grammar recognizes nonnegative decimal numbers,
`a-b` / `a to b` (optionally `years old`), `a+` / `a and above`.
At least three nonoverlapping numeric intervals are required. Order is sorted
by numeric lower bounds, independent of declared choice order; rank is scaled
to [0,1]. No midpoint or extrapolated upper bound is invented. Only `Not
answered`, `[generalised:sN]` and the declared gated value may be skipped as
sentinels. Unknown text, numbered nominal prefixes, overlap or duplicates
reject the entire added channel. Sentinel/absent answers map to numeric NaN
and retain their original categorical channel. Two-valued wave is rejected.
Schemas do not formally declare numeric types; numeric-label interpretation is
an explicit testable assumption, especially for bare codes.

CatBoost supports a mixed DataFrame with explicit categorical-column indices,
and numeric NaN is handled by its default missing-value mechanism:
[fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit),
[missing values](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing).
Actual pinned-backend mixed/NaN behavior must additionally pass a new native
control; documentation alone is not execution proof.

## Unchanged competition contract

All R11 CatBoost parameters, model seed 20261012, 96 iterations/depth6, >=90
visible labels, one fit per eligible target, pseudocount0.5 and complete support
remain unchanged. Full visible target-specific training rows, all original
GIVEN categorical columns, full queries and canonical output ordering remain.
No target-as-feature, row ID/role/index, query-statistic fitting, target hard
gates, partial scoring, OOF fits, files/weights/network calls. Upstream
CatBoost/ordinal encoding are existing techniques, not claimed inventions.

## Frozen complete paired experiment

Fresh seeds `202610181`, `202610182`; complete UNHCR -> UNICEF -> World Bank
in each cohort; frozen R11 executed on exactly the same newly generated full
inputs. Exact unmodified official scorer/schema/config/commit above. At most
12 scores / 6 pairs. Original promotion gate: >=4/6 deltas >=0.005, none below
-0.015. Stop immediately on material regression, unreachable required wins,
execution/resource failure. C2 requires an actually complete permissible C1.
No rescue via parser/feature/seed/hyperparameter/gate changes after results,
no choosing favourable items or counting speed alone as progress. If closed,
preserve the negative result and keep R11 as the valid selected candidate.

Public standard free Linux x86_64 Python3.12 CPU only. R17 predict caps
600/90/150=840 seconds with 60 seconds startup margin, parent sum <=900 per
cohort; unchanged R11 comparator caps480/120/240. Each complete cohort <=1800s,
job35min, output50MiB, >=10GiB free before work; max two conditional sequential
cohorts. Dependency versions exactly match R11 and official data tooling.
New tests run once on C1, including strict parser, mixed-feature routing,
sentinel/query independence and real pinned backend controls. Old completed
tests/jobs/collectors are not rerun as progress.

Only if complete original quality gate passes: a separately frozen real
phase2 full-contract runtime gate and exact ZIP validation must precede one
permitted Development upload. Test is not selected by this protocol. Synthetic
quality, official Development score, Test result and award remain distinct.
