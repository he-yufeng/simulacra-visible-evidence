"""Twelve new GIVEN-only density representation controls, not event scores."""
import ast
import copy
from importlib import metadata
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "participant_r18"))
spec = importlib.util.spec_from_file_location("given_density_r18", ROOT / "participant_r18/main.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class RecordingBackend:
    fits = []
    queries = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, frame, labels, cat_features):
        self.classes_ = np.unique(labels)
        self.fits.append({"frame": frame.copy(deep=True), "labels": np.array(labels),
                          "cat_features": list(cat_features), "parameters": self.parameters})
        return self

    def predict_proba(self, frame):
        self.queries.append(frame.copy(deep=True))
        return np.full((len(frame), len(self.classes_)), 1 / len(self.classes_))


class GivenDensityTests(unittest.TestCase):
    def setUp(self):
        RecordingBackend.fits, RecordingBackend.queries = [], []

    def fixture(self):
        choices = ["0", "1-2", "3-4", "5-6", "7+", "Not answered"]
        frame = pd.DataFrame({"a": choices * 41, "b": ["left", "right"] * 123,
                              "p": [0, 0, 0, 1, 1, 1] * 40 + [np.nan] * 6,
                              "respondent_id": np.arange(246),
                              "role": ["TRAIN"] * 240 + ["DEV"] * 6})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": choices},
            "b": {"class": "GIVEN", "values": ["left", "right"]},
            "p": {"class": "PREDICT", "values": [0, 1, 2]},
            "respondent_id": {"class": "EXCLUDE", "values": []}}}
        return frame, schema

    def test_original_r11_parameters_helpers_dependencies_and_support_unchanged(self):
        def selected(path):
            return {getattr(node, "name", None) or node.targets[0].id: ast.dump(node)
                    for node in ast.parse(path.read_text()).body
                    if (isinstance(node, ast.FunctionDef) and node.name in ("_layout", "_prior", "_expand"))
                    or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id in ("PARAMETERS", "MIN_LABELS", "PSEUDOCOUNT", "RESERVED_COLUMNS"))}
        self.assertEqual(selected(ROOT / "participant_r11/main.py"),
                         selected(ROOT / "participant_r18/main.py"))
        for name in ("requirements.txt", "LICENSE"):
            self.assertEqual((ROOT / "participant_r11" / name).read_bytes(),
                             (ROOT / "participant_r18" / name).read_bytes())

    def test_original_owned_density_math_and_constants_are_frozen(self):
        names = {"_prior", "_softmax", "_initial_responsibilities", "_maximization",
                 "_responsibilities", "_fit_latent"}
        def selected(path):
            return {getattr(node, "name", None) or node.targets[0].id: ast.dump(node)
                    for node in ast.parse(path.read_text()).body
                    if (isinstance(node, ast.FunctionDef) and node.name in names)
                    or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name))}
        self.assertEqual(selected(ROOT / "participant_r09/latent_core.py"),
                         selected(ROOT / "participant_r18/latent_given_core.py"))

    def test_encoder_fit_and_condition_receive_given_only_not_target_widths(self):
        frame, schema = self.fixture()
        with patch.object(agent, "_fit_latent", wraps=agent._fit_latent) as fitted:
            with patch.object(agent, "_responsibilities", wraps=agent._responsibilities) as conditioned:
                with patch.object(agent, "CatBoostClassifier", RecordingBackend):
                    agent.predict(frame, schema)
        codes, widths, train, given = fitted.call_args.args
        self.assertEqual(list(codes), ["a", "b"])
        self.assertEqual(widths, {"a": 6, "b": 2})
        self.assertEqual(given, ["a", "b"])
        self.assertEqual(train.tolist(), list(range(240)))
        self.assertEqual(list(conditioned.call_args.args[0]), ["a", "b"])
        self.assertEqual(conditioned.call_args.args[1], ["a", "b"])

    def test_query_count_patterns_identifiers_and_index_cannot_change_encoder_fit(self):
        frame, schema = self.fixture()
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            first = agent.predict(frame, schema)
        first_fit = RecordingBackend.fits[0]["frame"]
        appended = frame.tail(6).copy()
        appended["a"], appended["b"] = "Not answered", "left"
        extended = pd.concat([frame, appended], ignore_index=True)
        extended["respondent_id"], extended["role"] = 999999, "IGNORED"
        extended.index = np.arange(9000, 9000 + len(extended))
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            second = agent.predict(extended, schema)
        pd.testing.assert_frame_equal(first_fit, RecordingBackend.fits[1]["frame"])
        pd.testing.assert_frame_equal(RecordingBackend.queries[0], RecordingBackend.queries[1].iloc[:6])
        self.assertEqual(first, second[:6])

    def test_target_label_values_do_not_enter_density_representation(self):
        frame, schema = self.fixture()
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            agent.predict(frame, schema)
        first = RecordingBackend.fits[0]["frame"]
        frame.loc[:239, "p"] = 1 - frame.loc[:239, "p"]
        schema["items"]["p"]["values"] = [0, 1, 2, 3, 4, 5]
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            agent.predict(frame, schema)
        pd.testing.assert_frame_equal(first, RecordingBackend.fits[1]["frame"])

    def test_all_original_categorical_channels_and_joint_posterior_preserved(self):
        frame, schema = self.fixture()
        frame.loc[0, "a"] = np.nan
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            agent.predict(frame, schema)
        fit = RecordingBackend.fits[0]
        self.assertEqual(fit["frame"]["g0"].iloc[:6].tolist(), ["missing", "v1", "v2", "v3", "v4", "v5"])
        self.assertEqual(fit["frame"]["g1"].iloc[:2].tolist(), ["v0", "v1"])
        self.assertEqual(list(fit["frame"]), ["g0", "g1", "l0", "l1", "l2", "l3"])
        posterior = fit["frame"].iloc[:, 2:].to_numpy()
        self.assertTrue(np.isfinite(posterior).all())
        self.assertTrue((posterior >= 0).all())
        np.testing.assert_allclose(posterior.sum(axis=1), 1, rtol=0, atol=1e-12)
        self.assertGreater(np.max(np.ptp(posterior, axis=0)), .01)
        self.assertEqual(fit["cat_features"], [0, 1])

    def test_full_visible_rows_every_target_and_input_immutability(self):
        frame, schema = self.fixture()
        frame["q"] = [0, 1] * 48 + [np.nan] * 150
        schema["items"]["q"] = {"class": "PREDICT", "values": [0, 1]}
        before, before_schema = frame.copy(deep=True), copy.deepcopy(schema)
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            vectors = agent.predict(frame, schema)
        self.assertEqual([len(f["labels"]) for f in RecordingBackend.fits], [240, 96])
        self.assertEqual(len(vectors), 156)
        for fit in RecordingBackend.fits:
            self.assertEqual(fit["parameters"], agent.PARAMETERS)
            self.assertEqual(fit["cat_features"], [0, 1])
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(schema, before_schema)

    def test_missing_sentinels_gated_positive_support_and_canonical_slots(self):
        frame, schema = self.fixture()
        frame.loc[[0, 5, 240, 245], "a"] = np.nan
        schema["items"]["a"]["gate"] = {"parent": "b"}
        frame.loc[1, "a"] = "NA_GATED"
        schema["items"]["p"]["gate"] = {"parent": "b"}
        schema["items"]["p"]["observed_if"] = "b == impossible"
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            vectors = agent.predict(frame, schema)
        slots = [(0, "a"), (5, "a"), (240, "a"), (240, "p"), (241, "p"),
                 (242, "p"), (243, "p"), (244, "p"), (245, "a"), (245, "p")]
        self.assertEqual(len(vectors), len(slots))
        self.assertEqual(RecordingBackend.fits[0]["frame"]["g0"].iloc[:2].tolist(), ["missing", "v6"])
        for vector, (_, name) in zip(vectors, slots):
            self.assertEqual(len(vector), 7 if name == "a" else 4)
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1)

    def test_empty_visible_no_given_constant_and_no_query_fallbacks(self):
        frame, schema = self.fixture()
        frame["p"] = np.nan
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            vectors = agent.predict(frame, schema)
        self.assertEqual(RecordingBackend.fits, [])
        np.testing.assert_allclose(vectors, np.full((246, 3), 1 / 3))
        frame, schema = self.fixture()
        frame["a"], frame["b"] = "0", "left"
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            vectors = agent.predict(frame, schema)
        self.assertEqual(RecordingBackend.fits, [])
        self.assertEqual(len(vectors), 6)
        schema["items"]["a"]["class"] = schema["items"]["b"]["class"] = "EXCLUDE"
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            self.assertEqual(agent.predict(frame, schema), vectors)
        frame["p"] = 0
        self.assertEqual(agent.predict(frame, schema), [])

    def test_real_pinned_mixed_backend_learns_complete_owned_control(self):
        self.assertEqual(metadata.version("catboost"), "1.2.10")
        frame, schema = self.fixture()
        frame.loc[[0, 5, 240, 245], "a"] = np.nan
        before, before_schema = frame.copy(deep=True), copy.deepcopy(schema)
        vectors = agent.predict(frame, schema)
        slots = [(row, name) for row in range(len(frame))
                 for name, item in schema["items"].items()
                 if item["class"] in ("GIVEN", "PREDICT") and pd.isna(frame[name].iloc[row])]
        self.assertEqual(len(vectors), 10)
        self.assertEqual(len(vectors), len(slots))
        targets = [vector for vector, (_, name) in zip(vectors, slots) if name == "p"]
        self.assertEqual(len(targets), 6)
        for vector, truth in zip(targets[1:5], [0, 0, 1, 1]):
            self.assertGreater(vector[truth], .7)
        for vector in vectors:
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(schema, before_schema)

    def test_bad_visible_support_and_reserved_schema_fail_closed(self):
        frame, schema = self.fixture()
        frame.loc[0, "a"] = "unknown"
        with self.assertRaisesRegex(ValueError, "outside schema"):
            agent.predict(frame, schema)
        frame, schema = self.fixture()
        schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaisesRegex(ValueError, "identifier"):
            agent.predict(frame, schema)

    def test_owned_payload_has_no_external_io_or_unknown_imports(self):
        allowed = {"numpy", "pandas", "catboost", "latent_given_core"}
        for name in ("main.py", "latent_given_core.py"):
            tree = ast.parse((ROOT / "participant_r18" / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(a.name in allowed for a in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, allowed)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"open", "print", "input", "eval", "exec", "__import__"})


if __name__ == "__main__":
    unittest.main()
