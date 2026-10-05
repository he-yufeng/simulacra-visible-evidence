"""Visible-only per-target categorical boosting with complete option support.

Copyright (c) 2026 Yufeng He. Schema conventions follow MIT SituatedEvals.
CatBoost is an upstream Apache-2.0 dependency, not an invented algorithm.
No identifiers, target-as-feature, hard gates, networks, files or weights.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

MIN_LABELS = 90
PSEUDOCOUNT = 0.5
PARAMETERS = dict(iterations=96, depth=6, learning_rate=0.08, l2_leaf_reg=5.0,
    random_seed=20261012, loss_function="MultiClass", boosting_type="Ordered",
    bootstrap_type="Bayesian", bagging_temperature=1.0, one_hot_max_size=16,
    max_ctr_complexity=1, random_strength=1.0, thread_count=2, task_type="CPU",
    use_best_model=False, allow_writing_files=False, verbose=False)
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
    features = pd.DataFrame({f"g{index}": ["missing" if value < 0 else f"v{value}"
                                        for value in codes[name]]
                             for index, name in enumerate(given)})
    for name in targets:
        if name not in outputs:
            continue
        train = np.flatnonzero(codes[name] >= 0)
        query = np.flatnonzero(codes[name] < 0)
        if not given or len(train) < MIN_LABELS or len(np.unique(codes[name][train])) < 2:
            continue
        if not any(features.iloc[train][column].nunique() > 1 for column in features):
            continue
        model = CatBoostClassifier(**PARAMETERS)
        model.fit(features.iloc[train], codes[name][train], cat_features=list(range(len(given))))
        probabilities = model.predict_proba(features.iloc[query])
        if len(probabilities) != len(query):
            raise ValueError("backend query count differs")
        outputs[name][query] = _expand(probabilities, model.classes_, len(options[name]), len(train))
    return [outputs[name][row].tolist()
            for row in range(len(frame)) for name in items if codes[name][row] < 0]
