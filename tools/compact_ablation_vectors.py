"""Lossless float64 transport for owned synthetic ablation vectors."""
from itertools import chain
import numpy as np

VARIANTS = ("r05", "r09", "always_half")


def pack(path, outputs):
    if set(outputs) != set(VARIANTS):
        raise ValueError("unexpected variants")
    widths = np.array([len(vector) for vector in outputs["r05"]], dtype=np.int64)
    if np.any(widths <= 0):
        raise ValueError("empty vector support")
    arrays = {"widths": widths}
    for variant in VARIANTS:
        vectors = outputs[variant]
        if len(vectors) != len(widths) or any(len(vector) != width for vector, width in zip(vectors, widths)):
            raise ValueError("noncanonical counts or widths")
        arrays[variant] = np.fromiter(chain.from_iterable(vectors), dtype=np.float64,
                                      count=int(widths.sum()))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def unpack(path, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(VARIANTS) | {"widths"}:
            raise ValueError("unexpected archive keys")
        widths, values = bundle["widths"], bundle[variant]
    if (widths.dtype != np.dtype("int64") or values.dtype != np.dtype("float64")
            or widths.ndim != 1 or values.ndim != 1 or np.any(widths <= 0)
            or int(widths.sum()) != values.size):
        raise ValueError("invalid compact vector layout")
    offsets = np.concatenate(([0], np.cumsum(widths)))
    return [values[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
