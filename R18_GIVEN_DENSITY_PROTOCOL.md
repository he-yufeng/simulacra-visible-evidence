# R18: TRAIN-GIVEN-only density posterior beside all R11 categorical channels

Frozen before any R18 questionnaire-quality measurement, 2026-10-06 UTC+8.

## Independent mechanism and limitations

The exact public sampler at SituatedEvals/public
`0d2332d8ae19a8ce171031142bdc97134910e7ec` shifts multiple items using a
shared unobserved trait. This motivates testing a representation of GIVEN
co-occurrence; it does not establish the mechanism of private real data.
R03 already tried selected pair-count models; they are not revived here.
R17 numerical order produced no useful first-cohort gains and remains closed.

Reuse the participant-owned R08 finite-mixture density mathematics unchanged,
but fit it on GIVEN columns ONLY in rows with at least one visible PREDICT
answer. PREDICT labels and option widths never enter the encoder. Hidden-query
rows do not participate in initialization, marginal tables, mixture masses or
EM. The original K<=12, rows-per-component60, EM24, prior20, initialization
scale8 and seed20261008 are frozen; no parameter rescue of the old R08 model.
Condition every row on its own GIVENs and append its posterior probabilities
as numeric columns beside ALL original R11 categorical GIVEN columns.

This is not R08 imputation, its failed probabilities, an expert mixture or a
revival of pure latent-class prediction. The complete R11 per-target CatBoost
heads remain responsible for every target. A plausible benefit is easier
access to joint demographic structure; collapse, duplication, overfitting and
runtime cost are plausible failures. No benefit is claimed before measurement.

Finite categorical mixtures/EM are established methods, not inventions:
[Linzer and Lewis (2011)](https://www.jstatsoft.org/article/view/v042i10).
The implementation is our own NumPy core, not copied/installed GPL poLCA code.
The [CatBoost fit API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit)
supports explicit categorical indices in a mixed DataFrame; actual pinned
backend behavior still requires a fresh native control.

## Unchanged prediction and privacy contract

CatBoost parameters/seed20261012/96 iterations/depth6, >=90 visible target
labels, pseudocount0.5, dependencies, full visible training rows, complete
queries and positive complete option support remain exact R11. No target as
feature, identifiers/role/index, query fitting, target hard gates, item-name or
country exceptions, weights/files/network/API calls. GIVEN sentinels stay
ordinary categorical values; absence is skipped in the density likelihood and
retains the original R11 `missing` categorical channel. No PREDICT-conditioned
latent posterior is supplied even on training rows.

## Frozen native and full paired gate

Twelve new native controls run once in C1: original source/parameter fidelity,
unchanged core mathematics, encoder excludes all PREDICT, query isolation,
label-value isolation, original categorical preservation, full-row/target
routing, missing/gated/canonical support, empty/constant fallbacks, native
mixed-backend learning, bad support/reserved columns, no-external-I/O source.
No skips, CPU fake replacing actual backend, favourable-case removal or
post-result threshold changes. Local work is AST/hash only below 10GiB free.

Fresh seeds `202610191`, `202610192`; full UNHCR -> UNICEF -> World Bank;
frozen R11 on exactly the same complete newly generated inputs. Exact public
scorer/schema/config/commit above, raw rather than noised scores. At most
12 scores / 6 pairs, >=4 deltas >=0.005 and none below -0.015. Stop immediately
on material regression, unreachable four wins or execution/resource failure.
C2 needs a complete permissible C1. Preserve failures; do not change models,
features, seeds, questionnaires, scoring denominators or gates to rescue it.

Standard free PUBLIC Linux x86_64 Python3.12 CPU. R18 predict caps600/90/150,
840 total +60 startup margin, parent<=900; R11 comparator480/120/240 unchanged.
Each complete cohort<=1800s, job35min, output50MiB, >=10GiB free before every
owned step. No paid runner/artifact/cache/cloud/GPU/API. At most two conditional
sequential cohorts. Four-file R18 ZIP includes the owned density core; actual
ZIP gate and extracted-source equality precede grading. New quality PASS
alone is not formal upload permission: real phase2 runtime and exact ZIP gates,
then current authenticated Development quota/SHA/rules must pass. No Test or
award claim; keep the existing official R11 Development0.62 when R18 closes.
