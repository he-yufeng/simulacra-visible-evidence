"""Eight new calibration/boundary controls; not questionnaire accuracy evidence."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "participant_r13"
sys.path.insert(0, str(FOLDER))
try:
    spec = importlib.util.spec_from_file_location("calibration_r13", FOLDER / "main.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
finally:
    sys.path.pop(0)


class CalibrationTests(unittest.TestCase):
    def fixture(self, n=300, queries=2):
        z = np.arange(n) % 2
        frame = pd.DataFrame({"given": z.tolist() + (np.arange(queries) % 2).tolist(),
            "p": z.tolist() + [np.nan] * queries,
            "q": ([0, 1, "NA_GATED"] * ((n + 2) // 3))[:n] + [np.nan] * queries,
            "respondent_id": np.arange(n + queries), "role": ["TRAIN"] * n + ["DEV"] * queries})
        schema = {"gated_value": "NA_GATED", "items": {
            "given": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]},
            "q": {"class": "PREDICT", "values": [0, 1], "gate": {"parent": "given"}},
            "respondent_id": {"class": "EXCLUDE", "values": []}}}
        return frame, schema

    def fake(self, frame, schema, uniform=False):
        outputs = []
        for row in range(len(frame)):
            for name, item in schema["items"].items():
                if item["class"] not in ("GIVEN", "PREDICT") or not pd.isna(frame[name].iloc[row]):
                    continue
                width = len(item["values"]) + bool(item.get("gate"))
                vector = np.full(width, 1 / width)
                if name == "p" and not uniform:
                    truth = int(frame["given"].iloc[row]) if pd.notna(frame["given"].iloc[row]) else 0
                    vector = np.array([.35, .35])
                    vector[truth] = .65
                outputs.append(vector.tolist())
        return outputs

    def test_entire_validation_block_masked_inputs_unchanged_and_core_frozen(self):
        frame, schema = self.fixture()
        before, old_schema = frame.copy(deep=True), copy.deepcopy(schema)
        with patch.object(agent.boost_core, "predict", side_effect=self.fake) as spy:
            agent.predict(frame, schema)
        self.assertEqual(spy.call_count, 2)
        masked = spy.call_args_list[0].args[0]
        validation = np.flatnonzero(np.arange(300) % 10 >= 7)
        self.assertTrue(masked.iloc[validation][["p", "q"]].isna().all().all())
        self.assertTrue(masked.iloc[300:][["p", "q"]].isna().all().all())
        pd.testing.assert_frame_equal(masked.drop(columns=["p", "q"]), before.drop(columns=["p", "q"]))
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(schema, old_schema)
        self.assertEqual(hashlib.sha256((FOLDER / "boost_core.py").read_bytes()).hexdigest(),
            "32f16501feeb7b6890262165d3f510b14b818208e0ee73aca716d847ead6cdf8")

    def test_only_helped_target_scaled_unselected_gate_and_given_exact(self):
        frame, schema = self.fixture()
        with patch.object(agent.boost_core, "predict", side_effect=self.fake):
            items, options, codes = agent.boost_core._layout(frame, schema)
            selected = agent._select(frame, schema, items, options, codes)
            actual = agent.predict(frame, schema)
        self.assertEqual(selected, {"p": .5})
        original = self.fake(frame, schema)
        self.assertGreater(actual[0][0], original[0][0])
        self.assertEqual(actual[1], original[1])
        self.assertEqual(len(actual[1]), 3)
        self.assertTrue(all(value > 0 for value in actual[1]))
        frame.loc[0, "given"] = np.nan
        with patch.object(agent.boost_core, "predict", side_effect=self.fake):
            actual = agent.predict(frame, schema)
        self.assertEqual(actual[0], self.fake(frame, schema)[0])

    def test_uniform_validation_returns_exact_original_list(self):
        frame, schema = self.fixture()
        original = self.fake(frame, schema, uniform=True)
        def backend(f, s):
            return self.fake(f, s, uniform=True) if f[["p", "q"]].isna().sum().max() > 2 else original
        with patch.object(agent.boost_core, "predict", side_effect=backend):
            self.assertIs(agent.predict(frame, schema), original)

    def test_queries_index_and_identifiers_do_not_change_selection(self):
        frame, schema = self.fixture(queries=2)
        extended, _ = self.fixture(queries=6)
        extended["respondent_id"] += 90000
        extended["role"] = "IGNORED"
        extended.index = np.arange(10000, 10000 + len(extended))
        with patch.object(agent.boost_core, "predict", side_effect=self.fake):
            first = agent.predict(frame, schema)
            many = agent.predict(extended, schema)
        self.assertEqual(many, first * 3)

    def test_partial_truth_excluded_small_sample_and_given_only_skip_calibration(self):
        frame, schema = self.fixture()
        frame.loc[200:299, "q"] = np.nan
        with patch.object(agent.boost_core, "predict", side_effect=self.fake):
            items, options, codes = agent.boost_core._layout(frame, schema)
            self.assertNotIn("q", agent._select(frame, schema, items, options, codes))
        for n in (0, 80, 299):
            small, small_schema = self.fixture(n=n)
            with patch.object(agent.boost_core, "predict", side_effect=self.fake) as spy:
                self.assertEqual(agent.predict(small, small_schema), self.fake(small, small_schema))
            self.assertEqual(spy.call_count, 1)
        complete, complete_schema = self.fixture(queries=0)
        complete.loc[0, "given"] = np.nan
        with patch.object(agent.boost_core, "predict", side_effect=self.fake) as spy:
            self.assertEqual(agent.predict(complete, complete_schema), self.fake(complete, complete_schema))
        self.assertEqual(spy.call_count, 1)

    def test_temperature_math_identity_and_complete_support(self):
        vector = np.array([.1, .2, .7])
        np.testing.assert_array_equal(agent._scaled(vector, 1), vector)
        np.testing.assert_allclose(agent._scaled(vector, .5), vector**2 / sum(vector**2))
        array = np.vstack([vector, vector[::-1]])
        actual = agent._scaled(array, 2)
        np.testing.assert_allclose(actual.sum(axis=1), [1, 1])
        self.assertTrue((actual > 0).all())

    def test_malformed_vectors_counts_unknown_values_and_reserved_fields_refused(self):
        frame, schema = self.fixture()
        for bad in ([np.nan, .5], [0, 1], [-.1, 1.1], [.1, .1], [.2, .3, .5]):
            with self.assertRaises(ValueError):
                agent._vector(bad, 2)
        with patch.object(agent.boost_core, "predict", return_value=[]):
            with self.assertRaises(ValueError):
                agent.predict(frame, schema)
        invalid = frame.copy()
        invalid.loc[0, "p"] = 99
        with self.assertRaises(ValueError):
            agent.predict(invalid, schema)
        invalid_schema = copy.deepcopy(schema)
        invalid_schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaises(ValueError):
            agent.predict(frame, invalid_schema)

    def test_actual_calibration_and_full_backend_keep96_rounds_and_valid_outputs(self):
        values = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (150, 1))
        frame = pd.DataFrame({"a": values[:, 0].tolist() + [0, 0, 1, 1],
            "b": values[:, 1].tolist() + [0, 1, 0, 1],
            "p": np.bitwise_xor(values[:, 0], values[:, 1]).tolist() + [np.nan] * 4})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": [0, 1]}, "b": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]}}}
        original = agent.boost_core.CatBoostClassifier
        models = []
        def factory(**parameters):
            model = original(**parameters)
            models.append(model)
            return model
        with patch.object(agent.boost_core, "CatBoostClassifier", side_effect=factory):
            vectors = agent.predict(frame, schema)
        self.assertEqual(len(models), 2)
        self.assertTrue(all(model.tree_count_ == 96 for model in models))
        for vector, truth in zip(vectors, [0, 1, 1, 0]):
            self.assertGreater(vector[truth], .75)
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1)


if __name__ == "__main__":
    unittest.main()
