"""Eight owned boundary/backend checks, not questionnaire accuracy evidence."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("categorical_r11", ROOT / "participant_r11/main.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class RecordingBackend:
    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, frame, labels, cat_features):
        self.classes_ = np.unique(labels)
        self.fits.append({"frame": frame.copy(deep=True), "labels": np.array(labels),
                          "categorical_features": list(cat_features), "parameters": self.parameters})
        return self

    def predict_proba(self, frame):
        return np.full((len(frame), len(self.classes_)), 1 / len(self.classes_))


class BoostTests(unittest.TestCase):
    def fixture(self):
        pairs = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (40, 1))
        values = np.bitwise_xor(pairs[:, 0], pairs[:, 1])
        frame = pd.DataFrame({"a": pairs[:, 0].tolist() + [0, 0, 1, 1],
                              "b": pairs[:, 1].tolist() + [0, 1, 0, 1],
                              "p": values.tolist() + [np.nan] * 4,
                              "respondent_id": np.arange(164),
                              "role": ["TRAIN"] * 160 + ["DEV"] * 4})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": [0, 1]},
            "b": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]},
            "respondent_id": {"class": "EXCLUDE", "values": []}}}
        return frame, schema

    def test_actual_nonlinear_backend_learns_controlled_xor(self):
        frame, schema = self.fixture()
        vectors = agent.predict(frame, schema)
        self.assertEqual(len(vectors), 4)
        for vector, truth in zip(vectors, [0, 1, 1, 0]):
            self.assertGreater(vector[truth], .75)
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1.0)

    def test_target_specific_visible_rows_and_given_only_features(self):
        frame, schema = self.fixture()
        frame["q"] = ([0, 1] * 48) + [np.nan] * 68
        schema["items"]["q"] = {"class": "PREDICT", "values": [0, 1]}
        before = frame.copy(deep=True)
        RecordingBackend.fits = []
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            agent.predict(frame, schema)
        self.assertEqual([len(entry["labels"]) for entry in RecordingBackend.fits], [160, 96])
        for entry in RecordingBackend.fits:
            self.assertEqual(list(entry["frame"]), ["g0", "g1"])
            self.assertEqual(entry["categorical_features"], [0, 1])
            self.assertEqual(entry["parameters"], agent.PARAMETERS)
            self.assertFalse(entry["parameters"]["allow_writing_files"])
            self.assertEqual(entry["parameters"]["task_type"], "CPU")
        pd.testing.assert_frame_equal(frame, before)

    def test_real_hidden_query_count_index_and_ids_do_not_change_predictions(self):
        frame, schema = self.fixture()
        first = agent.predict(frame, schema)
        extended = pd.concat([frame, frame.tail(4)], ignore_index=True)
        extended["respondent_id"] = np.arange(8000, 8000 + len(extended))
        extended["role"] = "IGNORED"
        extended.index = np.arange(10000, 10000 + len(extended))
        second = agent.predict(extended, schema)
        np.testing.assert_allclose(first, second[:4], rtol=0, atol=1e-12)
        np.testing.assert_allclose(first, second[4:], rtol=0, atol=1e-12)

    def test_nonconsecutive_reversed_classes_retain_unseen_support(self):
        frame = pd.DataFrame({"a": [0, 1] * 50 + [0], "p": [0, 2] * 50 + [np.nan]})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1, 2]}}}

        class ReverseBackend(RecordingBackend):
            def fit(self, x, y, cat_features):
                self.classes_ = np.array([2, 0])
                return self

            def predict_proba(self, x):
                return np.tile([.9, .1], (len(x), 1))

        with patch.object(agent, "CatBoostClassifier", ReverseBackend):
            vector = agent.predict(frame, schema)[0]
        np.testing.assert_allclose(vector, np.array([10.5, .5, 90.5]) / 101.5, rtol=0, atol=1e-15)

    def test_sparse_given_marginals_and_full_canonical_order(self):
        frame = pd.DataFrame({"a": [0, 1, 0, np.nan, 0, np.nan],
                              "p": [0, 1, 0, 1, np.nan, np.nan]})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]}}}
        before = frame.copy(deep=True)
        with patch.object(agent, "CatBoostClassifier", side_effect=AssertionError("sparse path must not fit")):
            self.assertEqual(agent.predict(frame, schema),
                             [[.625, .375], [.5, .5], [.625, .375], [.5, .5]])
        pd.testing.assert_frame_equal(frame, before)

    def test_soft_gate_is_regular_support_not_a_forced_answer(self):
        frame, schema = self.fixture()
        frame["p"] = frame["p"].astype(object)
        frame.loc[np.arange(0, 160, 7), "p"] = "NA_GATED"
        schema["items"]["p"]["gate"] = {"parent": "a"}
        schema["items"]["p"]["observed_if"] = "a == 1"
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            outputs = agent.predict(frame, schema)
        for vector in outputs:
            self.assertEqual(len(vector), 3)
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1.0)

    def test_invalid_input_backend_support_and_probabilities_fail_closed(self):
        frame, schema = self.fixture()
        frame.loc[0, "p"] = 9
        with self.assertRaisesRegex(ValueError, "outside schema"):
            agent.predict(frame, schema)
        frame, schema = self.fixture()
        schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaisesRegex(ValueError, "identifier"):
            agent.predict(frame, schema)
        for classes in ([0, 0], [0, 3], [0, 1.5]):
            with self.assertRaisesRegex(ValueError, "support"):
                agent._expand([[.5, .5]], classes, 3, 100)
        for values in ([[np.nan, .5]], [[-.1, 1.1]], [[.2, .2]]):
            with self.assertRaisesRegex(ValueError, "probabilities"):
                agent._expand(values, [0, 1], 2, 100)

    def test_empty_constant_single_class_and_complete_paths_skip_backend(self):
        frame, schema = self.fixture()
        with patch.object(agent, "CatBoostClassifier", side_effect=AssertionError("no useful fitting context")):
            self.assertEqual(agent.predict(frame.tail(4), schema), [[.5, .5]] * 4)
            self.assertEqual(agent.predict(frame.iloc[:160], schema), [])
            mono = frame.copy(deep=True)
            mono.loc[:159, "p"] = 0
            np.testing.assert_allclose(agent.predict(mono, schema),
                                       np.tile([160.5 / 161, .5 / 161], (4, 1)), rtol=0, atol=1e-15)
            constant = frame.copy(deep=True)
            constant.loc[:159, ["a", "b"]] = 0
            self.assertEqual(agent.predict(constant, schema), [[.5, .5]] * 4)
            no_given = frame[["p"]].copy()
            no_given_schema = {"gated_value": "NA_GATED", "items": {"p": schema["items"]["p"]}}
            self.assertEqual(agent.predict(no_given, no_given_schema), [[.5, .5]] * 4)


if __name__ == "__main__":
    unittest.main()
