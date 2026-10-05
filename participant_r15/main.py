"""One shared categorical MLP fit on visible labels and GIVEN-only features.

Copyright (c) 2026 Yufeng He. MLP, Adam and grouped softmax are standard
methods; this NumPy implementation does not export data or claim invention.
"""
import numpy as np
import pandas as pd

MIN_LABELS = 90
PSEUDOCOUNT = 0.5
HIDDEN = 64
STEPS = 192
LEARNING_RATE = 0.02
RIDGE = 0.001
PRIOR_MIX = 0.05
MODEL_SEED = 20261016
GRADIENT_CLIP = 5.0
MAX_WORKSPACE_BYTES = 768 * 1024**2
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


def _design(codes, options, given, rows):
    n = len(next(iter(codes.values())))
    width = sum(len(options[name]) + 1 for name in given)
    values = np.zeros((n, width), dtype=float)
    offset = 0
    for name in given:
        levels = len(options[name])
        indices = np.where(codes[name] >= 0, codes[name], levels)
        values[np.arange(n), offset + indices] = 1.0
        offset += levels + 1
    values -= values[rows].mean(axis=0)
    return values


def _log_softmax(logits, widths):
    widths = np.asarray(list(widths), dtype=np.int64)
    if (logits.ndim != 2 or not len(widths) or np.any(widths <= 0)
            or logits.shape[1] != int(widths.sum()) or not np.isfinite(logits).all()):
        raise ValueError("invalid grouped logits or support")
    starts = np.r_[0, np.cumsum(widths)[:-1]]
    maxima = np.maximum.reduceat(logits, starts, axis=1)
    shifted = logits - np.repeat(maxima, widths, axis=1)
    normalizers = np.add.reduceat(np.exp(shifted), starts, axis=1)
    return shifted - np.repeat(np.log(normalizers), widths, axis=1)


def _forward(design, parameters, widths):
    hidden = np.tanh(design @ parameters["W1"] + parameters["b1"])
    logp = _log_softmax(hidden @ parameters["W2"] + parameters["b2"], widths.values())
    return hidden, logp


def _loss_gradient(design, labels, widths, parameters):
    hidden, logp = _forward(design, parameters, widths)
    probabilities = np.exp(logp)
    residual, loss = np.zeros_like(probabilities), 0.0
    count_heads, offset = len(widths), 0
    for name, width in widths.items():
        target = np.asarray(labels[name])
        if (target.ndim != 1 or not np.issubdtype(target.dtype, np.integer)
                or len(target) != len(design) or np.any(target < -1) or np.any(target >= width)):
            raise ValueError("invalid masked training labels")
        rows = np.flatnonzero(target >= 0)
        if len(rows):
            scale = 1.0 / (count_heads * len(rows))
            loss -= float(logp[rows, offset + target[rows]].sum()) * scale
            block = probabilities[rows, offset:offset + width].copy()
            block[np.arange(len(rows)), target[rows]] -= 1.0
            residual[rows, offset:offset + width] = block * scale
        offset += width
    penalty = RIDGE / count_heads
    loss += .5 * penalty * (np.sum(parameters["W1"]**2) + np.sum(parameters["W2"]**2))
    back = (residual @ parameters["W2"].T) * (1 - hidden**2)
    gradients = {"W1": design.T @ back + penalty * parameters["W1"],
        "b1": back.sum(axis=0), "W2": hidden.T @ residual + penalty * parameters["W2"],
        "b2": residual.sum(axis=0)}
    if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
        raise ValueError("nonfinite objective or gradients")
    return float(loss), gradients


def _fit(design, labels, widths, priors):
    rng = np.random.default_rng(MODEL_SEED)
    total = sum(widths.values())
    parameters = {"W1": rng.normal(0, 1 / np.sqrt(design.shape[1]), (design.shape[1], HIDDEN)),
        "b1": rng.normal(0, .01, HIDDEN),
        "W2": rng.normal(0, .05 / np.sqrt(HIDDEN), (HIDDEN, total)),
        "b2": np.concatenate([np.log(priors[name]) for name in widths])}
    moments = {k: np.zeros_like(v) for k, v in parameters.items()}
    squares = {k: np.zeros_like(v) for k, v in parameters.items()}
    for step in range(1, STEPS + 1):
        _, gradient = _loss_gradient(design, labels, widths, parameters)
        norm = np.sqrt(sum(float(np.sum(value**2)) for value in gradient.values()))
        factor = min(1.0, GRADIENT_CLIP / max(norm, 1e-30))
        for name in parameters:
            g = gradient[name] * factor
            moments[name] = .9 * moments[name] + .1 * g
            squares[name] = .999 * squares[name] + .001 * g**2
            mean = moments[name] / (1 - .9**step)
            variance = squares[name] / (1 - .999**step)
            parameters[name] -= LEARNING_RATE * mean / (np.sqrt(variance) + 1e-8)
        if any(not np.isfinite(p).all() for p in parameters.values()):
            raise ValueError("nonfinite fitted network")
    return parameters


def _checked(vectors, width):
    values = np.asarray(vectors, dtype=float)
    if (values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all()
            or np.any(values <= 0) or not np.all(np.isclose(values.sum(axis=1), 1, rtol=0, atol=1e-6))):
        raise ValueError("invalid complete-support output")
    return values


def predict(frame, schema):
    items, options, codes = _layout(frame, schema)
    if not items or not any(np.any(codes[name] < 0) for name in items):
        return []
    given = [name for name in items if schema["items"][name]["class"] == "GIVEN"]
    targets = [name for name in items if schema["items"][name]["class"] == "PREDICT"]
    visible = np.zeros(len(frame), dtype=bool)
    for name in targets:
        visible |= codes[name] >= 0
    rows = np.flatnonzero(visible)
    priors = {name: _prior(codes[name][rows], len(options[name])) for name in items}
    outputs = {name: np.tile(priors[name], (len(frame), 1)) for name in items if np.any(codes[name] < 0)}
    active = [name for name in targets if int((codes[name][rows] >= 0).sum()) >= MIN_LABELS
              and len(np.unique(codes[name][rows][codes[name][rows] >= 0])) >= 2]
    if (given and any(name in outputs for name in active)
            and any(len(np.unique(codes[name][rows])) > 1 for name in given)):
        widths = {name: len(options[name]) for name in active}
        features, total = sum(len(options[name]) + 1 for name in given), sum(widths.values())
        parameters_size = features * HIDDEN + HIDDEN * total + HIDDEN + total
        estimate = 8 * (len(frame) * features + len(rows) * features + 6 * len(rows) * total
                        + 2 * len(frame) * total + 4 * len(rows) * HIDDEN + 6 * parameters_size)
        if estimate > MAX_WORKSPACE_BYTES:
            raise ValueError("full dense workspace exceeds declared cap; no thinning")
        design = _design(codes, options, given, rows)
        labels = {name: codes[name][rows] for name in active}
        parameters = _fit(design[rows], labels, widths, priors)
        _, logp = _forward(design, parameters, widths)
        probabilities, offset = np.exp(logp), 0
        for name, width in widths.items():
            if name in outputs:
                values = (1 - PRIOR_MIX) * probabilities[:, offset:offset + width] + PRIOR_MIX * priors[name]
                values /= values.sum(axis=1, keepdims=True)
                outputs[name] = _checked(values, width)
            offset += width
    return [outputs[name][row].tolist() for row in range(len(frame)) for name in items if codes[name][row] < 0]
