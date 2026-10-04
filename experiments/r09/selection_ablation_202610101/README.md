# Fresh-seed selector ablation: no supported incremental selector advantage

The unchanged r09 expert-selection method and an always-half mixture were
compared with frozen r05 on complete public synthetic Development frames from
seed 202610101, using the fixed official toolkit commit 0d2332d8ae19a8ce171031142bdc97134910e7ec.

All three r09-minus-r05 differences were positive. However, r09 exceeded the
always-half control on only one of three instruments; the mean difference
was -0.000064176. The predeclared requirement of at least two positive
instrument differences and a positive mean was NOT attained. Completion of
the nine score aggregates is not a pass for this mechanism hypothesis.

Nine grades shared captured real full-expert fits; they were not nine
independent predictor runs. The unchanged r09 function ran once per frame,
with full validation PREDICT blocks masked before both experts' fold fits.
Observer wrappers record calls without modifying decisions or returned
outputs. Frozen full-fit outputs supply r05 and the unconditional half control.
Per-item score reconciliation confirms selected targets match the half control,
while rejected targets match r05 exactly; the selector/control score difference
is therefore the rejected-target arithmetic contribution.

See summary.json for the actual full counts, raw skills, measured local times,
paired differences, source and immutable receipt hashes. Small differences
from one generator do not show population causality, significance or general
superiority. Do not automatically replace a formal candidate with an ablation
winner. Original source/ZIP, official best and Test selection are unchanged.

The initial completed-prediction attempt hit the 50MiB temporary JSON-output
gate before grading and retained zero scores. One explicit storage revision
used lossless float64 NPZ transport, verified by two owned transport checks;
the seed/model/comparison/resource limits were unchanged. Actual protocol and
resource revision documents accompany the public observation/transport helpers
and their controlled tests. These are not a new native contract or official
evaluation, and do not trigger the frozen source-path CI.

Only participant-owned code and public synthetic aggregate evidence are
published. No respondent answers, prediction arrays, real microdata, credentials,
private correspondence or CV are included. The report remains a private draft.
