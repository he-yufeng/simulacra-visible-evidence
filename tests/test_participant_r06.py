"""Owned microfixtures for the soft-route layer, not benchmark accuracy."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("soft_r06", ROOT / "participant_r06/main.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class SoftRouteContractTests(unittest.TestCase):
    def fixture(self, hidden_parent=False, queries=1):
        parent = np.array([0]*30 + [1]*30 + ([-1] if hidden_parent else [0])*queries)
        target = np.array([2]*25 + [0]*5 + [2]*5 + [1]*25 + [-1]*queries)
        codes = {"parent": parent, "child": target}
        options = {"parent": ["route", "skip"], "child": ["yes", "no", "NA_GATED"]}
        schema = {"gated_value": "NA_GATED", "items": {
            "parent": {"class": "PREDICT" if hidden_parent else "GIVEN"},
            "child": {"class": "PREDICT", "gate": {"parent": "parent", "observed_if": ["route"]}},
        }}
        probabilities = {"child": np.tile([.6, .2, .2], (len(parent), 1))}
        if hidden_parent:
            probabilities["parent"] = np.tile([.75, .25], (len(parent), 1))
        return probabilities, codes, options, schema

    def test_visible_parent_uses_learned_reliability_not_hard_gate(self):
        args = self.fixture()
        output = candidate._soft_route(*args)["child"]
        # Deliberate valid rule violations: parent='route' has25/30 gated children.
        # Shrunk reliability=(25+20*.5)/(30+20)=.7; blend gate=.45.
        np.testing.assert_allclose(output[-1], [.4125, .1375, .45])
        self.assertTrue(np.all(output > 0))
        np.testing.assert_allclose(output.sum(axis=1), 1)

    def test_hidden_predict_parent_is_integrated_not_read_as_truth(self):
        args = self.fixture(hidden_parent=True)
        output = candidate._soft_route(*args)["child"]
        # .75*.7 + .25*.3=.6; blend gate=.4.
        np.testing.assert_allclose(output[-1], [.45, .15, .4])

    def test_insufficient_complete_pairs_preserve_exact_original(self):
        predictions, codes, options, schema = self.fixture()
        codes["child"][0] = -1
        before = predictions["child"].copy()
        output = candidate._soft_route(predictions, codes, options, schema)
        np.testing.assert_array_equal(output["child"], before)

    def test_fully_masked_predict_rows_never_enter_training_pairs(self):
        one = candidate._soft_route(*self.fixture(hidden_parent=True, queries=1))["child"][-1]
        many = candidate._soft_route(*self.fixture(hidden_parent=True, queries=9))["child"][-9:]
        np.testing.assert_allclose(many, np.tile(one, (9, 1)))

    def test_original_parent_arrays_and_iteration_order_are_preserved(self):
        root = np.array([0]*30 + [1]*30 + [0])
        parent = np.array([0]*20 + [2]*10 + [1]*20 + [2]*10 + [-1])
        child = np.array([0]*10 + [2]*20 + [1]*10 + [2]*20 + [-1])
        codes = {"root": root, "parent": parent, "child": child}
        options = {"root": ["a", "b"], "parent": ["yes", "no", "NA_GATED"],
                   "child": ["yes", "no", "NA_GATED"]}
        schema = {"gated_value": "NA_GATED", "items": {
            "root": {"class": "GIVEN"},
            "parent": {"class": "PREDICT", "gate": {"parent": "root"}},
            "child": {"class": "PREDICT", "gate": {"parent": "parent"}},
        }}
        parent_p = np.tile([.6, .2, .2], (61, 1))
        child_p = np.tile([.2, .3, .5], (61, 1))
        before = {"parent": parent_p.copy(), "child": child_p.copy()}
        first = candidate._soft_route(dict(before), codes, options, schema)
        reversed_order = candidate._soft_route({"child": child_p, "parent": parent_p}, codes, options, schema)
        for name in before:
            np.testing.assert_allclose(first[name], reversed_order[name])
            np.testing.assert_array_equal(before[name], parent_p if name == "parent" else child_p)


if __name__ == "__main__":
    unittest.main()
