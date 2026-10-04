"""Temperature-calibrated multi-GIVEN evidence with nested r03 fallback.

Copyright (c) 2026 Yufeng He; schema conventions follow MIT SituatedEvals.
Local candidate, not submitted. Only visible labels and GIVEN features;
no identifiers, hard routing, files, network, model weights or logging.
"""
from itertools import combinations

import numpy as np
import pandas as pd

PRIOR_MASS = 20.0
MIN_LABELS = 60
MIN_NESTED_LABELS = 90
MAX_FEATURE_LEVELS = 64
MAX_PAIR_LEVELS = 256
MIN_LOG_GAIN = 0.002
SE_MULTIPLIER = 2.0
FOLDS = 3
SCORING_FLOOR = 0.001
FEATURE_COUNTS = (1, 2, 4, 8)
TEMPERATURES = (0.25, 0.5, 1.0)
MAX_EVIDENCE_FEATURES = 8


def _prior(labels, width):
    counts = np.bincount(labels, minlength=width).astype(float) + 0.5
    return counts / counts.sum()


def _conditional(labels, feature, levels, width, prior):
    valid = feature >= 0
    counts = np.bincount(feature[valid] * width + labels[valid],
                         minlength=levels * width).reshape(levels, width)
    return (counts + PRIOR_MASS * prior) / (counts.sum(axis=1, keepdims=True) + PRIOR_MASS)


def _lookup(table, feature, prior):
    probabilities = np.tile(prior, (len(feature), 1))
    known = feature >= 0
    probabilities[known] = table[feature[known]]
    return probabilities


def _select_feature(y, width, features):
    """Same fixed three-fold selection heuristic as the r02 fallback."""
    labelled = np.flatnonzero(y >= 0)
    if len(labelled) < MIN_LABELS or len(np.unique(y[labelled])) < 2:
        return None
    folds = np.arange(len(labelled)) % FOLDS
    gains = {name: np.zeros(len(labelled), dtype=float) for name in features}
    for fold in range(FOLDS):
        train = labelled[folds != fold]
        valid = labelled[folds == fold]
        prior = _prior(y[train], width)
        reference = np.log(np.maximum(prior[y[valid]], 1e-3))
        for name, (values, levels) in features.items():
            table = _conditional(y[train], values[train], levels, width, prior)
            probabilities = _lookup(table, values[valid], prior)
            logp = np.log(np.maximum(probabilities[np.arange(len(valid)), y[valid]], 1e-3))
            gains[name][folds == fold] = logp - reference
    best, best_gain = None, 0.0
    for name, gain in gains.items():
        mean = float(gain.mean())
        error = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        if mean > max(MIN_LOG_GAIN, SE_MULTIPLIER * error) and mean > best_gain:
            best, best_gain = name, mean
    return best


def _pair_features(features):
    pairs = {}
    for left, right in combinations(features, 2):
        x, nx = features[left]
        z, nz = features[right]
        if nx * nz > MAX_PAIR_LEVELS:
            continue
        known = (x >= 0) & (z >= 0)
        values = np.full(len(x), -1, dtype=np.int64)
        values[known] = x[known] * nz + z[known]
        pairs[(left, right)] = (values, nx * nz)
    return pairs


def _model_probabilities(y, width, features, selected, rows):
    labels = y >= 0
    prior = _prior(y[labels], width)
    if selected is None:
        return np.tile(prior, (len(rows), 1))
    values, levels = features[selected]
    table = _conditional(y[labels], values[labels], levels, width, prior)
    return _lookup(table, values[rows], prior)


def _nested_pair_gain(y, width, singles, pairs):
    """Outer validation rows never affect either inner model selection or fit.

    This estimates an algorithm's gain, not a formal confidence interval. Both
    selections are rerun inside each training fold. No PREDICT block features.
    """
    labelled = np.flatnonzero(y >= 0)
    folds = np.arange(len(labelled)) % FOLDS
    gain = np.zeros(len(labelled), dtype=float)
    for fold in range(FOLDS):
        valid = labelled[folds == fold]
        train_y = y.copy()
        train_y[valid] = -1
        single = _select_feature(train_y, width, singles)
        pair = _select_feature(train_y, width, pairs)
        base = _model_probabilities(train_y, width, singles, single, valid)
        candidate = _model_probabilities(train_y, width, pairs, pair, valid)
        index = np.arange(len(valid))
        scale = 1.0 - width * SCORING_FLOOR
        base_p = SCORING_FLOOR + scale * base[index, y[valid]]
        pair_p = SCORING_FLOOR + scale * candidate[index, y[valid]]
        gain[folds == fold] = np.log(pair_p) - np.log(base_p)
    return gain


def _select_model(y, width, singles, pairs):
    single = _select_feature(y, width, singles)
    labelled = np.flatnonzero(y >= 0)
    if (not pairs or len(labelled) < MIN_NESTED_LABELS
            or len(np.unique(y[labelled])) < 2):
        return singles, single
    gain = _nested_pair_gain(y, width, singles, pairs)
    mean = float(gain.mean())
    error = float(gain.std(ddof=1) / np.sqrt(len(gain)))
    if mean > max(MIN_LOG_GAIN, SE_MULTIPLIER * error):
        pair = _select_feature(y, width, pairs)
        if pair is not None:
            return pairs, pair
    return singles, single


def _rank_features(y, width, singles):
    """Rank useful features using only the supplied visible training labels."""
    labelled = np.flatnonzero(y >= 0)
    if len(labelled) < MIN_LABELS or len(np.unique(y[labelled])) < 2:
        return []
    folds = np.arange(len(labelled)) % FOLDS
    gains = {name: np.zeros(len(labelled), dtype=float) for name in singles}
    scale = 1.0 - width * SCORING_FLOOR
    for fold in range(FOLDS):
        train, valid = labelled[folds != fold], labelled[folds == fold]
        prior = _prior(y[train], width)
        reference = np.log(SCORING_FLOOR + scale * prior[y[valid]])
        for name, (values, levels) in singles.items():
            table = _conditional(y[train], values[train], levels, width, prior)
            probabilities = _lookup(table, values[valid], prior)
            logp = np.log(SCORING_FLOOR + scale * probabilities[np.arange(len(valid)), y[valid]])
            gains[name][folds == fold] = logp - reference
    ranked = []
    for name, gain in gains.items():
        mean = float(gain.mean())
        error = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        if mean > max(MIN_LOG_GAIN, SE_MULTIPLIER * error):
            ranked.append((name, mean))
    ranked.sort(key=lambda item: -item[1])
    return [name for name, _ in ranked[:MAX_EVIDENCE_FEATURES]]


def _aggregate_evidence(y, width, singles, ranked, rows):
    """Compute each conditional table once, then cache prefix evidence sums."""
    labelled = y >= 0
    prior = _prior(y[labelled], width)
    total = np.zeros((len(rows), width), dtype=float)
    prefixes = {}
    for index, name in enumerate(ranked[:MAX_EVIDENCE_FEATURES], start=1):
        values, levels = singles[name]
        table = _conditional(y[labelled], values[labelled], levels, width, prior)
        known = values[rows] >= 0
        total[known] += np.log(table[values[rows][known]]) - np.log(prior)
        if index in FEATURE_COUNTS:
            prefixes[index] = total.copy()
    for count in FEATURE_COUNTS:
        if count not in prefixes:
            prefixes[count] = total.copy()
    return prior, prefixes


def _temperature_probabilities(prior, evidence, temperature):
    logits = np.log(prior) + temperature * evidence
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.maximum(np.exp(logits), np.finfo(float).tiny)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _evidence_probabilities(y, width, singles, ranked, configuration, rows):
    prior, prefixes = _aggregate_evidence(y, width, singles, ranked, rows)
    if configuration is None:
        return np.tile(prior, (len(rows), 1))
    count, temperature = configuration
    return _temperature_probabilities(prior, prefixes[count], temperature)


def _select_evidence_configuration(y, width, singles):
    labelled = np.flatnonzero(y >= 0)
    if len(labelled) < MIN_LABELS or len(np.unique(y[labelled])) < 2:
        return None
    configurations = [(count, temperature) for count in FEATURE_COUNTS for temperature in TEMPERATURES]
    folds = np.arange(len(labelled)) % FOLDS
    gains = {configuration: np.zeros(len(labelled), dtype=float) for configuration in configurations}
    scale = 1.0 - width * SCORING_FLOOR
    for fold in range(FOLDS):
        valid = labelled[folds == fold]
        train_y = y.copy()
        train_y[valid] = -1
        ranked = _rank_features(train_y, width, singles)
        prior, prefixes = _aggregate_evidence(train_y, width, singles, ranked, valid)
        reference = np.log(SCORING_FLOOR + scale * prior[y[valid]])
        for configuration in configurations:
            count, temperature = configuration
            probabilities = _temperature_probabilities(prior, prefixes[count], temperature)
            logp = np.log(SCORING_FLOOR + scale * probabilities[np.arange(len(valid)), y[valid]])
            gains[configuration][folds == fold] = logp - reference
    best, best_gain = None, 0.0
    for configuration, gain in gains.items():
        mean = float(gain.mean())
        error = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        if mean > max(MIN_LOG_GAIN, SE_MULTIPLIER * error) and mean > best_gain:
            best, best_gain = configuration, mean
    return best


def _nested_evidence_gain(y, width, singles, pairs):
    """Both complete selection algorithms are trained inside each outer fold."""
    labelled = np.flatnonzero(y >= 0)
    folds = np.arange(len(labelled)) % FOLDS
    gain = np.zeros(len(labelled), dtype=float)
    scale = 1.0 - width * SCORING_FLOOR
    for fold in range(FOLDS):
        valid = labelled[folds == fold]
        train_y = y.copy()
        train_y[valid] = -1
        base_features, base_selected = _select_model(train_y, width, singles, pairs)
        configuration = _select_evidence_configuration(train_y, width, singles)
        ranked = _rank_features(train_y, width, singles) if configuration is not None else []
        base = _model_probabilities(train_y, width, base_features, base_selected, valid)
        candidate = _evidence_probabilities(train_y, width, singles, ranked, configuration, valid)
        index = np.arange(len(valid))
        base_p = SCORING_FLOOR + scale * base[index, y[valid]]
        candidate_p = SCORING_FLOOR + scale * candidate[index, y[valid]]
        gain[folds == fold] = np.log(candidate_p) - np.log(base_p)
    return gain


def _choose_evidence_model(y, width, singles, pairs):
    labelled = np.flatnonzero(y >= 0)
    if (singles and len(labelled) >= MIN_NESTED_LABELS
            and len(np.unique(y[labelled])) >= 2):
        gain = _nested_evidence_gain(y, width, singles, pairs)
        mean = float(gain.mean())
        error = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        if mean > max(MIN_LOG_GAIN, SE_MULTIPLIER * error):
            configuration = _select_evidence_configuration(y, width, singles)
            if configuration is not None:
                return "evidence", (_rank_features(y, width, singles), configuration)
    return "r03", _select_model(y, width, singles, pairs)


def predict(frame, schema):
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    options = {name: list(schema["items"][name]["values"]) +
               ([schema["gated_value"]] if schema["items"][name].get("gate") else [])
               for name in items}
    codes = {}
    for name in items:
        if np.any(~frame[name].isna().to_numpy() & ~frame[name].isin(options[name]).to_numpy()):
            raise ValueError("answer outside declared option support")
        codes[name] = pd.Categorical(frame[name], categories=options[name]).codes.astype(np.int64)
    singles = {name: (codes[name], len(options[name])) for name in items
               if schema["items"][name]["class"] == "GIVEN"
               and len(options[name]) <= MAX_FEATURE_LEVELS}
    pairs = _pair_features(singles)
    predictions = {}
    rows = np.arange(len(frame))
    for name in items:
        y = codes[name]
        if not np.any(y < 0):
            continue
        if schema["items"][name]["class"] == "PREDICT":
            method, selected = _choose_evidence_model(y, len(options[name]), singles, pairs)
            if method == "evidence":
                ranked, configuration = selected
                predictions[name] = _evidence_probabilities(
                    y, len(options[name]), singles, ranked, configuration, rows)
            else:
                features, feature = selected
                predictions[name] = _model_probabilities(
                    y, len(options[name]), features, feature, rows)
        else:
            predictions[name] = _model_probabilities(y, len(options[name]), singles, None, rows)
    return [predictions[name][row].tolist()
            for row in range(len(frame)) for name in items if codes[name][row] < 0]
