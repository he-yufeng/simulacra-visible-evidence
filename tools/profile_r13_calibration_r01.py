"""Correct a diagnostic entry guard with one original hidden-query sentinel."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time


def denied(*args, **kwargs):
    raise RuntimeError("owned diagnostic does not use network")


socket.socket = denied
socket.create_connection = denied
socket.getaddrinfo = denied

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "participant_r13"))
import boost_core
import main as calibration

EXPECTED = {
    "participant_r13/main.py": "1e9d461273c0043bb2810ef20fe5354757a653c3912c25f98a747678e0108d83",
    "participant_r13/boost_core.py": "32f16501feeb7b6890262165d3f510b14b818208e0ee73aca716d847ead6cdf8",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    started = time.monotonic()
    output = ROOT / "r13_calibration_profile_2026-10-05_r01.json"
    if output.exists():
        raise RuntimeError("one-shot diagnostic output exists; do not repeat")
    disk = __import__("os").statvfs(ROOT)
    if disk.f_bavail * disk.f_frsize < 10 * 1024**3:
        raise RuntimeError("stop below10GiB; no data cleanup authorized here")
    for name, expected in EXPECTED.items():
        assert sha(ROOT / name) == expected
    rng = np.random.default_rng(202610501)
    features = rng.integers(0, 8, size=(3600, 3))
    frame = pd.DataFrame({f"g{k}": features[:, k].astype(str) for k in range(3)})
    schema = {"gated_value": "NA_GATED", "items": {
        f"g{k}": {"class": "GIVEN", "values": [str(n) for n in range(8)]} for k in range(3)}}
    visible = np.arange(len(frame)) % 4 == 0
    for k in range(6):
        width = 3 + k % 2
        answers = (features[:, k % 3] + features[:, (k + 1) % 3] * (1 + k % 3)) % width
        noise = rng.random(len(frame)) < .12
        answers[noise] = rng.integers(0, width, size=int(noise.sum()))
        frame[f"p{k}"] = [str(v) if visible[row] else np.nan for row, v in enumerate(answers)]
        schema["items"][f"p{k}"] = {"class": "PREDICT", "values": [str(n) for n in range(width)]}
    frame_before = digest(frame.to_json())
    real_classifier, fits, phase = boost_core.CatBoostClassifier, [], "original"

    class ObservedClassifier:
        def __init__(self, **parameters):
            self.model = real_classifier(**parameters)
            self.record = {"phase": phase, "parameters_sha256": digest(parameters)}

        def fit(self, x, y, **kwargs):
            self.record.update(training_rows=len(x), training_input_sha256=digest({
                "features": x.to_numpy().tolist(), "labels": y.tolist(), "kwargs": kwargs}))
            tick = time.monotonic()
            self.model.fit(x, y, **kwargs)
            self.record["fit_seconds"] = time.monotonic() - tick
            fits.append(self.record)
            return self

        def predict_proba(self, x):
            tick = time.monotonic()
            result = self.model.predict_proba(x)
            self.record.update(prediction_rows=len(x), predict_seconds=time.monotonic() - tick)
            return result

        def __getattr__(self, name):
            return getattr(self.model, name)

    boost_core.CatBoostClassifier = ObservedClassifier
    try:
        items, options, codes = boost_core._layout(frame, schema)
        tick = time.monotonic()
        original_selection = calibration._select(frame, schema, items, options, codes)
        original_seconds = time.monotonic() - tick
        phase = "eligible_plus_one_query"
        eligible = np.flatnonzero(np.logical_or.reduce([codes[f"p{k}"] >= 0 for k in range(6)]))
        hidden = np.flatnonzero(~visible)
        # Original _select needs a missing target; retain one unchanged query.
        # This row never enters training. No model, split or grid changes.
        compact = frame.iloc[np.r_[eligible, hidden[:1]]].copy(deep=True)
        compact_items, compact_options, compact_codes = boost_core._layout(compact, schema)
        tick = time.monotonic()
        compact_selection = calibration._select(compact, schema, compact_items, compact_options, compact_codes)
        compact_seconds = time.monotonic() - tick
    finally:
        boost_core.CatBoostClassifier = real_classifier
    old, new = [f for f in fits if f["phase"] == "original"], [f for f in fits if f["phase"] == "eligible_plus_one_query"]
    print("PROFILE_STAGE_JSON " + json.dumps({"original_fits": len(old), "compact_fits": len(new),
        "original_seconds": original_seconds, "compact_seconds": compact_seconds}), flush=True)
    assert len(old) == len(new) == 6
    assert all(a["training_input_sha256"] == b["training_input_sha256"] and
               a["parameters_sha256"] == b["parameters_sha256"] for a, b in zip(old, new))
    assert original_selection == compact_selection
    assert digest(frame.to_json()) == frame_before
    assert all(sha(ROOT / name) == expected for name, expected in EXPECTED.items())
    assert time.monotonic() - started < 60
    result = {"scope": "ONE_OWNED_FIXTURE_ORIGINAL_VS_ELIGIBLE_PLUS_ONE_QUERY_DIAGNOSIS_NOT_FULL_SURVEY",
        "status": "PASS", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": EXPECTED, "invented_seed": 202610501, "total_rows": 3600,
        "visible_rows": int(visible.sum()), "fully_hidden_original_query_rows": int((~visible).sum()),
        "targets": 6, "retained_hidden_query_rows_for_needed_guard": 1, "compact_total_rows": len(compact), "actual_96_round_fits": len(fits), "fits": fits,
        "original_selection": original_selection, "eligible_plus_one_query_selection": compact_selection,
        "selections_identical": True, "per_target_training_values_labels_and_parameters_identical": True,
        "original_selection_seconds": original_seconds, "eligible_plus_one_query_selection_seconds": compact_seconds,
        "wall_seconds": time.monotonic() - started, "input_and_original_source_unchanged": True,
        "full_baseline_refit_or_final_predictions_executed": False,
        "official_survey_seed_data_grader_or_terminal_cohort_reexecuted": False,
        "inner_r13_failure_root_cause_confirmed": False,
        "quality_shared900_or_actual_full_entry_contract_verified": False,
        "new_participant_revision_published_or_formally_uploaded": False,
        "network_calls": 0, "new_paid_service_or_commitment_cny": 0,
        "limits": "One small invented six-target fixture. Equal selection and training digests are mechanism evidence only, not universal probability equivalence, repeated timing statistics, full-questionnaire quality,600/900s feasibility or original630s failure diagnosis. Query-row removal changes calibration workload only; original full-data baseline refit and all final queries would still be required in a future implementation."}
    encoded = json.dumps(result, indent=2) + "\n"
    assert len(encoded.encode()) < 1024**2
    with output.open("x") as stream:
        stream.write(encoded)
    print(json.dumps({k: result[k] for k in ("status", "actual_96_round_fits", "selections_identical",
        "original_selection_seconds", "eligible_plus_one_query_selection_seconds", "wall_seconds")}))


if __name__ == "__main__":
    main()
