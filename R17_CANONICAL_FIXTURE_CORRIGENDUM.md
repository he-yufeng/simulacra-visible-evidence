# R17 canonical-output fixture corrigendum

Recorded before any R17 questionnaire-quality measurement, 2026-10-06 UTC+8.

The original public run 37366916303 attempt 1 never acquired a runner. After
GitHub Actions was reported operational, attempt 2 acquired runner 1000007429
and job 112012177346 failed in the new native controls, before packaging,
dataset generation, fitting either questionnaire model, or scoring any cell.
Eleven controls passed; the real mixed-backend control failed an unchanged
probability-above-0.7 assertion. The raw failure and original test are preserved.

The fixture makes GIVEN `a` missing at rows 0, 5, 240 and 245 and target `p`
missing at rows 240-245. The participant contract returns every missing GIVEN
and PREDICT in row/item order, including TRAIN rows. Thus the ten output slots
are `(0,a), (5,a), (240,a), (240,p), (241,p), (242,p), (243,p), (244,p),
(245,a), (245,p)`. The old hard-coded indices selected six slots under the
incorrect assumption of only two missing GIVENs; the first learning assertion
tested marginal `(240,a)` probability 0.16390041493775934 as if it were a
target prediction. This establishes a fixture-index error, not model quality.

The correction derives missing slots from the original input and schema,
asserts the exact ten-slot ordering/count, and selects only the six `p`
vectors. The same four targets at rows 241-244 must still exceed 0.7 for
truth `[0,0,1,1]`. All original inputs, NaN locations, twelve test cases,
finite/positive/normalized output checks and the 0.7 threshold remain.
No participant code, feature grammar, CatBoost parameter, dependency, seed,
scoring input or original six-pair promotion/resource gate is changed.

A single corrected-fixture C1 run is permitted to validate this correction
and then execute the original full questionnaire protocol if controls pass.
Do not erase the failed attempt, lower thresholds, skip cases or interpret
fixture correction as a questionnaire gain. C2 remains conditional on the
original complete C1 rules; phase 2 and official upload remain separate gates.
