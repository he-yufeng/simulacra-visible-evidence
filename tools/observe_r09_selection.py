"""Owned non-mutating capture for a fixed selection ablation, not a solver."""
from contextlib import ExitStack
import time
from unittest.mock import patch


def capture(agent, frame, schema):
    """Run actual r09 once; never substitute cached full fits into validation."""
    full, calls, selections = {}, [], []
    originals = {"r05": agent.evidence_core.predict, "latent": agent.latent_core.predict}
    select = agent._select_mixtures

    def wrap(name):
        def observed(seen_frame, seen_schema):
            started = time.monotonic()
            output = originals[name](seen_frame, seen_schema)
            is_full = seen_frame is frame
            calls.append({"expert": name, "full_fit": is_full,
                          "wall_seconds": time.monotonic() - started})
            if is_full:
                if name in full:
                    raise ValueError("repeated full fit unexpectedly encountered")
                full[name] = output
            return output
        return observed

    def observe_selector(*args):
        chosen = select(*args)
        selections.append(set(chosen))
        return chosen

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent.evidence_core, "predict", wrap("r05")))
        stack.enter_context(patch.object(agent.latent_core, "predict", wrap("latent")))
        stack.enter_context(patch.object(agent, "_select_mixtures", observe_selector))
        started = time.monotonic()
        actual = agent.predict(frame, schema)
        elapsed = time.monotonic() - started
    if len(selections) != 1 or "r05" not in full:
        raise ValueError("selector/full baseline capture missing")
    extra_latent = "latent" not in full
    if extra_latent:
        started = time.monotonic()
        full["latent"] = originals["latent"](frame, schema)
        calls.append({"expert": "latent", "full_fit": True, "ablation_only": True,
                      "wall_seconds": time.monotonic() - started})
    return {"r09": actual, "r05": full["r05"], "latent": full["latent"],
            "selected": selections[0], "calls": calls,
            "r09_wall_seconds": elapsed, "extra_full_latent_for_ablation": extra_latent}


def always_half(base, latent, names, predict_names):
    """Fixed convex control; no labels or target-selection decisions accepted."""
    if len(base) != len(latent) or len(base) != len(names):
        raise ValueError("noncanonical expert output counts")
    output = []
    for b, alternative, name in zip(base, latent, names):
        if name not in predict_names:
            output.append(b)
            continue
        if len(b) != len(alternative):
            raise ValueError("expert vector widths differ")
        combined = [(.5 * float(x) + .5 * float(y)) for x, y in zip(b, alternative)]
        total = sum(combined)
        if total <= 0:
            raise ValueError("zero mixture mass")
        output.append([value / total for value in combined])
    return output
