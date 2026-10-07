# R20: joint dependence-tree estimation and conditional sum-product

New generative structure, not R19 backend/tree-parameter rescue or R08 mixture
revival. Fit ALL schema GIVEN+PREDICT nodes on rows with at least one visible
PREDICT. Every variable/pair is retained; jointcounts use only mutually visible
answers. Whole hidden-query rows enter neither counts, priors nor structure.
Allowed visible PREDICT labels are joint-density observations, not leaked query
truth/discriminative features. Condition each output row on exactly its supplied
visible answers, marginalizing every missing variable. IDs/role/EXCLUDE/index,
question/country exceptions, observed_if hard rules, API/network/files absent.

Regularized available-case Chow-Liu-style fit: each pair >=90 observations uses
complete-support counts+.5 for MI; lower-count pairs have weight0. Maximum
spanning tree by deterministic Prim, root schemaindex0, ties first schemaorder.
Positive transitions counts+.5 per state; sparse pairs use learned child
marginals as independent conditionals. Root/all fallback marginals counts+.5.
These choices precede first scores. Not a guaranteed exact complete-data KL
MLE with missingness/smoothing, causal discovery, or a claimed new algorithm.

Owned NumPy/SciPy log-space sum-product gives exact conditional marginals of
THIS learned positive tree, up to float64/1e-300 underflow guard; not exact
real-world probabilities. All queries retained;128-row batching bounds memory,
does not downsample. Gate sentinel is an ordinary observed category, NaN means
unobserved. Pairwise tree cannot learn pure XOR: explicit negative native
control demonstrates this limitation rather than claiming universal capacity.
Other native12 include independent exhaustive joint enumeration, all-row
batching, actual chain fit, query/training isolation, canonical/support/softgate/
partial-label and extreme likelihood behavior. No skips or fake-backend stand-in.

Unchanged stronger R11 source32f165/ZIPa53d and public SituatedEvals0d2332d8
scorer/config/full schemas. Fresh202610211/202610212, FULL UNHCR->UNICEF->WB,
same paired inputs/fullcounts. Original>=4of6 delta>=+.005/no delta<-.015 gate;
resource/contract failure, materialregression or unreachablewins closes family.
Only complete attainable C1 permits one C2. No tuning seeds, graphroot/prior/
batch/model choices after outcomes, dropped variables/items, originalfailed
R19 reuse, favourable subset or unknown-score-as-zero. Fullsixmean null unless
all6pairs complete. R11Oct5 official.62 remains datedbest, no new officialscore.

Predict capsBOTH480/120/240=840+60startup, each sum of actualparent scores<=900;
cohort1800s/job35min/output50MiB/remote>=10GiBdisk. Existing PUBLIC standard
Ubuntu24/Python3.12 CPU free, no artifact/cache/larger runner/GPU/modelAPI/
Alibaba/paidcommit or thirddeep. Pin numpy2.5.3/pandas3.0.6/scipy1.18.1 and
same comparatorcatboost1.2.10. Local lowdisk: source/AST only, no scientific
imports/installs. Source public is owned generic implementation/synthetic
fixtures only; confidential real data/private outputs/credentials never exported.

After actualterminal: obtain complete originalreceipt; independent source/ZIP/
schema/weightedlogscore/pair/plan audit, then close onlynewworkflow on final
failure/pass. QualityPASS alone is not uploadpermission: exactphase2runtime,
currentauthenticatedDevquota/rules/SHA still needed. No earlyTest or prize claim.
Monthly2000/planning1550 and existingprivateL2D20commit/actualunknown untouched.

References: established dependence trees (Chow/Liu1968, DOI10.1109/TIT.1968.1054142)
and sum-product (Kschischang/Frey/Loeliger2001). Direct primary fulltexts did not
load in this check; do not claim a fullpaperread. Inference correctness here is
tested by independent enumeration of explicitly defined positive tiny trees.
https://technav.ieee.org/topic/sum-product-algorithm/
https://github.com/SituatedEvals/public/tree/0d2332d8ae19a8ce171031142bdc97134910e7ec
