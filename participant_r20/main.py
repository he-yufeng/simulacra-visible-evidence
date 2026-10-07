"""Visible-training joint density, conditioned only on supplied answers.

Copyright (c)2026 Yufeng He. Schema boundary/prior helpers are owned R11.
Dependence trees/message passing are established methods, not inventions.
No identifiers, query fit, hard gate, network, files, or hidden labels.
"""
import numpy as np
import pandas as pd
from tree_core import fit_tree, infer_tree

PSEUDOCOUNT = .5
RESERVED_COLUMNS = {"respondent_id","role"}


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


def predict(frame, schema):
    items, options, codes = _layout(frame,schema)
    if not items or not any(np.any(value<0) for value in codes.values()):
        return []
    targets = [name for name in items if schema["items"][name]["class"]=="PREDICT"]
    visible = np.zeros(len(frame),dtype=bool)
    for name in targets:
        visible |= codes[name]>=0
    data = np.column_stack([codes[name] for name in items])
    widths = [len(options[name]) for name in items]
    tree = fit_tree(data[visible],widths)
    missing_rows = np.flatnonzero(np.any(data<0,axis=1))
    posterior = infer_tree(tree,data[missing_rows])
    return [posterior[index][position].tolist()
            for position,row in enumerate(missing_rows)
            for index,name in enumerate(items) if codes[name][row]<0]
