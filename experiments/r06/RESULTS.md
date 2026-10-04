# Closed fixed experiment: soft learned routing

Completed run37173691214 at commit2e4049870469aa8e3c73dca897b2a18e3660da01.
All six completed scoring calls and archive gates passed. The method gate did
not: three gains were positive but too small. STOP_FUTILITY was taken exactly
when zero useful completed pairs plus three remaining pairs could not reach
four. No second-seed cells, final-stage replay or official r06 upload followed.

| Instrument | r05 raw synthetic skill | r06 raw synthetic skill | Delta |
|---|---:|---:|---:|
| UNHCR | 0.524735 | 0.526020 | +0.001284 |
| UNICEF | 0.573473 | 0.574394 | +0.000921 |
| World Bank | 0.449165 | 0.449627 | +0.000462 |

These are invented-data results, not the official displayed r05 Development
score0.58. The full aggregate receipt records sources, process/dataset hashes,
denominators and raw versus noised/reported measures; no raw respondent rows
or confidential data are published. Log-score/schema-uniform-reference
recomputation and paired counts agree within floating-point precision.
The first180-second baseline timeout is retained separately in the workflow
history and first receipt. It is neither a r06 regression nor an official
r05 failure.

This experiment is closed. Do not tune its blend/seed/threshold and rerun it
to manufacture a favorable result. A distinct future method needs a new
protocol and bounded resource plan.
