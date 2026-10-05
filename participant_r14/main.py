"""Runtime repair of R13: eligible-only calibration, batched vector handling.

Copyright (c) 2026 Yufeng He. The R11 learner, full-query refit, held-out
selection rule and complete schema support are unchanged. No network or IDs.
"""
import numpy as np
import boost_core

MIN_VISIBLE = 300
MIN_VALIDATION_LABELS = 90
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MIN_LOG_GAIN = 0.002
SE_MULTIPLIER = 2.0
FLOOR = 0.001


def _scaled(vectors, temperature):
    if temperature == 1.0:
        return vectors.copy()
    logp = np.log(vectors) / temperature
    values = np.exp(logp - np.max(logp, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def _groups(frame, schema, vectors):
    """Validate canonical outputs in target batches, without per-cell dicts."""
    items, options, codes = boost_core._layout(frame, schema)
    if not items:
        if len(vectors):
            raise ValueError("outputs without schema items")
        return {}
    missing = np.column_stack([codes[name] < 0 for name in items])
    rows, columns = np.nonzero(missing)
    if len(vectors) != len(rows):
        raise ValueError("backend output count differs from canonical missing cells")
    groups = {}
    for column, name in enumerate(items):
        indices = np.flatnonzero(columns == column)
        if not len(indices):
            continue
        values = np.asarray([vectors[int(index)] for index in indices], dtype=float)
        if (values.shape != (len(indices), len(options[name]))
                or not np.isfinite(values).all() or np.any(values <= 0)
                or not np.all(np.isclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6))):
            raise ValueError("invalid complete-support backend probabilities")
        groups[name] = (indices, rows[indices], values)
    return groups


def _select(frame, schema, items, options, codes):
    targets = [name for name in items if schema["items"][name]["class"] == "PREDICT"]
    # Needed is fixed on ORIGINAL input, not inferred from the compact frame.
    needed = [name for name in targets if np.any(codes[name] < 0)]
    if not needed:
        return {}
    visible = np.zeros(len(frame), dtype=bool)
    for name in targets:
        visible |= codes[name] >= 0
    eligible = np.flatnonzero(visible)
    if len(eligible) < MIN_VISIBLE:
        return {}
    # Remove only fully-hidden original queries from the calibration pass.
    # They supply no training label or validation truth. Final pass retains all.
    validation = np.flatnonzero(np.arange(len(eligible)) % 10 >= 7)
    masked = frame.iloc[eligible].copy(deep=True)
    for name in targets:
        masked.iloc[validation, masked.columns.get_loc(name)] = np.nan
    groups = _groups(masked, schema, boost_core.predict(masked, schema))
    selected = {}
    for name in needed:
        rows = validation[codes[name][eligible[validation]] >= 0]
        if len(rows) < MIN_VALIDATION_LABELS:
            continue
        _, predicted_rows, probabilities = groups[name]
        locations = np.searchsorted(predicted_rows, rows)
        if np.any(locations >= len(predicted_rows)) or not np.array_equal(predicted_rows[locations], rows):
            raise ValueError("validation predictions not canonically aligned")
        probabilities = probabilities[locations]
        labels = codes[name][eligible[rows]]
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
        if best_gain > max(MIN_LOG_GAIN, SE_MULTIPLIER * best_error):
            selected[name] = best_temperature
    return selected


def predict(frame, schema):
    items, options, codes = boost_core._layout(frame, schema)
    selected = _select(frame, schema, items, options, codes)
    # Full original input, all visible labels, all original queries: mandatory.
    baseline = boost_core.predict(frame, schema)
    if not selected:
        return baseline
    groups = _groups(frame, schema, baseline)
    result = list(baseline)
    for name, temperature in selected.items():
        indices, _, probabilities = groups[name]
        transformed = _scaled(probabilities, temperature).tolist()
        for index, vector in zip(indices, transformed):
            result[int(index)] = vector
    return result
