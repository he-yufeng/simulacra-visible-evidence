"""Schema-onehot extremely randomized tree probability estimation.

Copyright (c) 2026 Yufeng He. Owned R11 support/layout helpers preserved.
ExtraTrees is established scikit-learn BSD-3 software, not an invented method.
All GIVENs; target-specific visible labels; no identifiers/query fit/I/O.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

MIN_LABELS = 90
PSEUDOCOUNT = 0.5
PARAMETERS = dict(n_estimators=64, criterion="log_loss", max_depth=12,
    min_samples_leaf=8, min_samples_split=2, max_features="sqrt",
    bootstrap=False, class_weight=None, random_state=20261020, n_jobs=2)
RESERVED_COLUMNS = {"respondent_id", "role"}


def _layout(frame, schema):
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    if RESERVED_COLUMNS.intersection(items):
        raise ValueError("identifier cannot be a predictive variable")
    options, codes = {}, {}
    for name in items:
        choices = list(schema["items"][name]["values"])
        if schema["items"][name].get("gate"):
            choices.append(schema["gated_value"])
        if not choices:
            raise ValueError("empty option support")
        if np.any(~frame[name].isna().to_numpy() & ~frame[name].isin(choices).to_numpy()):
            raise ValueError("visible answer outside schema support")
        options[name] = choices
        codes[name] = pd.Categorical(frame[name], categories=choices).codes.astype(np.int64)
    return items, options, codes


def _prior(labels, width):
    counts = np.bincount(labels[labels >= 0], minlength=width).astype(float) + PSEUDOCOUNT
    return counts / counts.sum()


def _expand(probabilities, classes, width, labels):
    """Refuse malformed backends; retain unseen and schema-gated options."""
    p = np.asarray(probabilities, dtype=float)
    values = np.asarray(classes)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all()):
        raise ValueError("invalid backend class support")
    ids = values.astype(np.int64)
    if (not np.array_equal(ids, values) or len(set(ids.tolist())) != len(ids)
            or np.any(ids < 0) or np.any(ids >= width)):
        raise ValueError("invalid backend class support")
    if (p.ndim != 2 or p.shape[1] != len(ids) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("invalid backend probabilities")
    full = np.zeros((len(p), width), dtype=float)
    full[:, ids] = p / p.sum(axis=1, keepdims=True)
    full = (labels * full + PSEUDOCOUNT) / (labels + PSEUDOCOUNT * width)
    return full / full.sum(axis=1, keepdims=True)


def _features(given, options, codes, rows):
    """Complete schema onehot and explicit missing slot, no fitted statistics."""
    width = sum(len(options[name])+1 for name in given)
    matrix = np.zeros((rows, width), dtype=np.float32)
    offsets, offset = {}, 0
    indices = np.arange(rows)
    for name in given:
        slots = len(options[name])
        offsets[name] = (offset, slots+1)
        positions = np.where(codes[name] < 0, slots, codes[name])
        matrix[indices, offset+positions] = 1
        offset += slots+1
    return matrix, offsets


def predict(frame, schema):
    items, options, codes = _layout(frame, schema)
    if not items or not any(np.any(values < 0) for values in codes.values()):
        return []
    given = [name for name in items if schema["items"][name]["class"] == "GIVEN"]
    targets = [name for name in items if schema["items"][name]["class"] == "PREDICT"]
    visible = np.zeros(len(frame), dtype=bool)
    for name in targets:
        visible |= codes[name] >= 0
    priors = {name: _prior(codes[name][visible], len(options[name])) for name in items}
    outputs = {name: np.tile(priors[name], (len(frame), 1))
               for name in items if np.any(codes[name] < 0)}
    features, offsets = _features(given, options, codes, len(frame))
    for name in targets:
        if name not in outputs:
            continue
        train = np.flatnonzero(codes[name] >= 0)
        query = np.flatnonzero(codes[name] < 0)
        if not given or len(train) < MIN_LABELS or len(np.unique(codes[name][train])) < 2:
            continue
        if not np.any(np.ptp(features[train], axis=0) > 0):
            continue
        model = ExtraTreesClassifier(**PARAMETERS)
        model.fit(features[train], codes[name][train])
        probabilities = model.predict_proba(features[query])
        if len(probabilities) != len(query):
            raise ValueError("backend query count differs")
        outputs[name][query] = _expand(probabilities, model.classes_, len(options[name]), len(train))
    return [outputs[name][row].tolist()
            for row in range(len(frame)) for name in items if codes[name][row] < 0]
