"""Whole-PREDICT-block cross-fitted fixed expert mixture, visible data only.

Copyright (c) 2026 Yufeng He. The two bundled, exact frozen cores are owned
MIT code. Validation labels train only this selector, never either expert.
"""
import numpy as np
import pandas as pd

import evidence_core
import latent_core

FOLDS = 3
MIN_LABELS = 90
MIN_LOG_GAIN = 0.002
SE_MULTIPLIER = 2.0
FLOOR = 0.001
MIX_WEIGHT = 0.5
RESERVED_COLUMNS = {"respondent_id", "role"}


def _layout(frame, schema):
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    if RESERVED_COLUMNS.intersection(items):
        raise ValueError("identifier cannot be a model feature")
    options = {name: list(schema["items"][name]["values"]) +
               ([schema["gated_value"]] if schema["items"][name].get("gate") else [])
               for name in items}
    codes = {}
    for name in items:
        if not options[name]:
            raise ValueError("empty declared option support")
        if np.any(~frame[name].isna().to_numpy() & ~frame[name].isin(options[name]).to_numpy()):
            raise ValueError("answer outside declared option support")
        codes[name] = pd.Categorical(frame[name], categories=options[name]).codes.astype(np.int64)
    return items, options, codes


def _vector(value, width):
    vector = np.asarray(value, dtype=float)
    if (vector.shape != (width,) or not np.isfinite(vector).all()
            or np.any(vector < 0) or not np.isclose(vector.sum(), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("invalid expert probability vector")
    return vector / vector.sum()


def _decoded(frame, schema, vectors):
    items, options, codes = _layout(frame, schema)
    cells = [(row, name) for row in range(len(frame)) for name in items if codes[name][row] < 0]
    if len(vectors) != len(cells):
        raise ValueError("expert output count does not match canonical missing cells")
    return {cell: _vector(vector, len(options[cell[1]])) for cell, vector in zip(cells, vectors)}


def _select_mixtures(frame, schema, items, options, codes):
    predicted = [name for name in items if schema["items"][name]["class"] == "PREDICT"]
    eligible = np.zeros(len(frame), dtype=bool)
    for name in predicted:
        eligible |= codes[name] >= 0
    labelled = np.flatnonzero(eligible)
    if len(labelled) < MIN_LABELS:
        return set()
    columns = [frame.columns.get_loc(name) for name in predicted]
    fold_ids = np.arange(len(labelled)) % FOLDS
    gains = {name: np.full(len(labelled), np.nan) for name in predicted}
    for fold in range(FOLDS):
        validation_positions = np.flatnonzero(fold_ids == fold)
        validation_rows = labelled[validation_positions]
        masked = frame.copy(deep=True)
        # Explicit object promotion permits NaN even for fully visible integer
        # columns. Positional masking never treats the dataframe index as a feature.
        for name in predicted:
            masked[name] = masked[name].astype(object)
        masked.iloc[validation_rows, columns] = np.nan
        base = _decoded(masked, schema, evidence_core.predict(masked, schema))
        latent = _decoded(masked, schema, latent_core.predict(masked, schema))
        for position, row in zip(validation_positions, validation_rows):
            for name in predicted:
                truth = codes[name][row]
                if truth < 0:
                    continue
                base_probability = base[(row, name)][truth]
                combined = (1 - MIX_WEIGHT) * base_probability + MIX_WEIGHT * latent[(row, name)][truth]
                scale = 1 - len(options[name]) * FLOOR
                gains[name][position] = np.log(FLOOR + scale * combined) - np.log(FLOOR + scale * base_probability)
    selected = set()
    for name, differences in gains.items():
        observed = differences[np.isfinite(differences)]
        if len(observed) < MIN_LABELS:
            continue
        standard_error = float(observed.std(ddof=1) / np.sqrt(len(observed)))
        if float(observed.mean()) > max(MIN_LOG_GAIN, SE_MULTIPLIER * standard_error):
            selected.add(name)
    return selected


def predict(frame, schema):
    items, options, codes = _layout(frame, schema)
    if not any(np.any(values < 0) for values in codes.values()):
        return []
    selected = _select_mixtures(frame, schema, items, options, codes)
    original = evidence_core.predict(frame, schema)
    if not selected:
        return original
    alternative = latent_core.predict(frame, schema)
    cells = [(row, name) for row in range(len(frame)) for name in items if codes[name][row] < 0]
    if len(original) != len(cells) or len(alternative) != len(cells):
        raise ValueError("expert output count does not match canonical missing cells")
    output = []
    for (_, name), base, latent in zip(cells, original, alternative):
        base_vector = _vector(base, len(options[name]))
        latent_vector = _vector(latent, len(options[name]))
        if name in selected:
            combined = (1 - MIX_WEIGHT) * base_vector + MIX_WEIGHT * latent_vector
            output.append((combined / combined.sum()).tolist())
        else:
            output.append(base)
    return output
