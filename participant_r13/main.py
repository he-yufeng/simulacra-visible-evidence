"""Whole-block-held-out per-target temperature calibration of frozen R11.

Copyright (c) 2026 Yufeng He. Temperature scaling is established calibration,
not an invented learner. No hidden labels, identifiers, hard gates or network.
"""
import numpy as np
import boost_core

MIN_VISIBLE = 300
MIN_VALIDATION_LABELS = 90
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MIN_LOG_GAIN = 0.002
SE_MULTIPLIER = 2.0
FLOOR = 0.001


def _vector(value, width):
    vector = np.asarray(value, dtype=float)
    if (vector.shape != (width,) or not np.isfinite(vector).all() or np.any(vector <= 0)
            or not np.isclose(vector.sum(), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("invalid complete-support probability vector")
    return vector


def _scaled(vector, temperature):
    if temperature == 1.0:
        return vector.copy()
    logp = np.log(vector) / temperature
    values = np.exp(logp - np.max(logp, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def _decoded(frame, schema, vectors):
    items, options, codes = boost_core._layout(frame, schema)
    cells = [(row, name) for row in range(len(frame)) for name in items if codes[name][row] < 0]
    if len(vectors) != len(cells):
        raise ValueError("baseline output count differs from canonical missing cells")
    return {cell: _vector(vector, len(options[cell[1]])) for cell, vector in zip(cells, vectors)}


def _select(frame, schema, items, options, codes):
    targets = [name for name in items if schema["items"][name]["class"] == "PREDICT"]
    needed = [name for name in targets if np.any(codes[name] < 0)]
    if not needed:
        return {}
    visible = np.zeros(len(frame), dtype=bool)
    for name in targets:
        visible |= codes[name] >= 0
    eligible = np.flatnonzero(visible)
    if len(eligible) < MIN_VISIBLE:
        return {}
    # Positions only split available labels; never enter the predictor as features.
    validation = eligible[np.arange(len(eligible)) % 10 >= 7]
    masked = frame.copy(deep=True)
    for name in targets:
        masked.iloc[validation, masked.columns.get_loc(name)] = np.nan
    predictions = _decoded(masked, schema, boost_core.predict(masked, schema))
    selected = {}
    for name in needed:
        rows = validation[codes[name][validation] >= 0]
        if len(rows) < MIN_VALIDATION_LABELS:
            continue
        probabilities = np.vstack([predictions[row, name] for row in rows])
        labels = codes[name][rows]
        width = len(options[name])
        baseline = np.log(FLOOR + (1 - width * FLOOR) * probabilities[np.arange(len(rows)), labels])
        best_temperature, best_gain, best_error = 1.0, 0.0, 0.0
        for temperature in TEMPERATURES:
            if temperature == 1.0:
                continue
            scaled = _scaled(probabilities, temperature)
            difference = np.log(FLOOR + (1 - width * FLOOR) * scaled[np.arange(len(rows)), labels]) - baseline
            gain = float(difference.mean())
            if gain > best_gain:
                best_temperature, best_gain = temperature, gain
                best_error = float(difference.std(ddof=1) / np.sqrt(len(rows)))
        # Fixed selection heuristic, not a confidence interval or significance test.
        if best_gain > max(MIN_LOG_GAIN, SE_MULTIPLIER * best_error):
            selected[name] = best_temperature
    return selected


def predict(frame, schema):
    items, options, codes = boost_core._layout(frame, schema)
    selected = _select(frame, schema, items, options, codes)
    baseline = boost_core.predict(frame, schema)
    if not selected:
        return baseline
    cells = [(row, name) for row in range(len(frame)) for name in items if codes[name][row] < 0]
    if len(cells) != len(baseline):
        raise ValueError("full baseline output count differs")
    result = []
    for (row, name), original in zip(cells, baseline):
        vector = _vector(original, len(options[name]))
        result.append(_scaled(vector, selected[name]).tolist() if name in selected else original)
    return result
