"""Shared categorical latent density; visible fit and GIVEN-only conditioning.

Copyright (c) 2026 Yufeng He; schema conventions follow MIT SituatedEvals.
Independent NumPy implementation of a standard finite mixture idea. No
identifiers, external files/network/weights, hard gates or private-data logs.
"""
import numpy as np
import pandas as pd

MAX_COMPONENTS = 12
ROWS_PER_COMPONENT = 60
EM_STEPS = 24
PRIOR_MASS = 20.0
INIT_SCALE = 8.0
INITIALIZATION_SEED = 20261008
RESERVED_COLUMNS = {"respondent_id", "role"}


def _prior(labels, width):
    counts = np.bincount(labels, minlength=width).astype(float) + 0.5
    return counts / counts.sum()


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _initial_responsibilities(codes, train, given_names, components):
    rng = np.random.default_rng(INITIALIZATION_SEED)
    order = rng.permutation(len(train))
    anchors, seen = [], set()
    for position in order:
        profile = tuple(int(codes[name][train[position]]) for name in given_names)
        if profile not in seen:
            anchors.append(int(train[position]))
            seen.add(profile)
        if len(anchors) == components:
            break
    if len(anchors) < components:
        for position in order:
            row = int(train[position])
            if row not in anchors:
                anchors.append(row)
            if len(anchors) == components:
                break
    matches = np.zeros((len(train), components), dtype=float)
    observed = np.zeros_like(matches)
    for name in given_names:
        values = codes[name][train, None]
        anchor_values = codes[name][anchors][None, :]
        valid = (values >= 0) & (anchor_values >= 0)
        matches += valid & (values == anchor_values)
        observed += valid
    return _softmax(INIT_SCALE * matches / np.maximum(observed, 1.0))


def _maximization(codes, widths, train, responsibilities, priors):
    components = responsibilities.shape[1]
    masses = responsibilities.sum(axis=0)
    mixture = (masses + 0.5) / (len(train) + 0.5 * components)
    tables = {}
    for name, width in widths.items():
        values = codes[name][train]
        valid = values >= 0
        counts = np.stack([np.bincount(values[valid],
            weights=responsibilities[valid, component], minlength=width)
            for component in range(components)])
        tables[name] = (counts + PRIOR_MASS * priors[name]) / (
            counts.sum(axis=1, keepdims=True) + PRIOR_MASS)
    return mixture, tables


def _responsibilities(codes, names, rows, mixture, tables):
    logits = np.tile(np.log(mixture), (len(rows), 1))
    for name in names:
        values = codes[name][rows]
        valid = values >= 0
        if np.any(valid):
            logits[valid] += np.log(tables[name][:, values[valid]].T)
    return _softmax(logits)


def _fit_latent(codes, widths, train, given_names):
    priors = {name: _prior(codes[name][train][codes[name][train] >= 0], width)
              for name, width in widths.items()}
    components = min(MAX_COMPONENTS, max(1, len(train) // ROWS_PER_COMPONENT))
    if components == 1 or not given_names:
        return np.ones(1), {name: prior[None, :] for name, prior in priors.items()}
    responsibilities = _initial_responsibilities(codes, train, given_names, components)
    for _ in range(EM_STEPS):
        mixture, tables = _maximization(codes, widths, train, responsibilities, priors)
        responsibilities = _responsibilities(codes, list(widths), train, mixture, tables)
    return mixture, tables
