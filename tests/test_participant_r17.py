"""Twelve new schema-order and mixed-backend controls, not event scores."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("schema_order_r17", ROOT / "participant_r17/main.py")
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


class SchemaOrderTests(unittest.TestCase):
    def setUp(self):
        RecordingBackend.fits, RecordingBackend.queries = [], []

    def fixture(self):
        choices = ["0", "1-2", "3-4", "5-6", "7+", "Not answered"]
        frame = pd.DataFrame({"a": choices * 40 + choices,
                              "b": ["left", "right"] * 123,
                              "p": [0, 0, 0, 1, 1, 1] * 40 + [np.nan] * 6,
                              "respondent_id": np.arange(246),
                              "role": ["TRAIN"] * 240 + ["DEV"] * 6})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": choices},
            "b": {"class": "GIVEN", "values": ["left", "right"]},
            "p": {"class": "PREDICT", "values": [0, 1, 2]},
            "respondent_id": {"class": "EXCLUDE", "values": []}}}
        return frame, schema

    def test_whole_label_grammar_not_numbered_nominal_prefix(self):
        for value, expected in (("0", (0., 0.)), ("1-1999", (1., 1999.)),
                ("18 to 24 years old", (18., 24.)), ("25 - 34 years old", (25., 34.)),
                ("35\u201344", (35., 44.)), ("1.5 to 2.5", (1.5, 2.5)),
                ("25000+", (25000., np.inf)), ("45 and above", (45., np.inf))):
            self.assertEqual(agent._numeric_interval(value), expected)
        for value in ("1. Head of household", "1. No", "yes1", "3 years old maybe",
                      "nan", "inf", "1e9", "-1", "5-3", "1-" + "9" * 400):
            self.assertIsNone(agent._numeric_interval(value), value)

    def test_sorted_disjoint_coordinates_not_declaration_order_or_midpoints(self):
        result = agent._ordered_coordinates(["25000+", "0", "Not answered", "1-1999",
                                             "[generalised:s2]", "NA_GATED"], "NA_GATED")
        np.testing.assert_allclose(result, [1, 0, np.nan, .5, np.nan, np.nan], equal_nan=True)
        self.assertEqual(agent._ordered_coordinates(["0", "1", "2", "4", "6-10", "11+"], None).tolist(),
                         [0, .2, .4, .6, .8, 1])

    def test_ambiguous_partial_overlapping_and_binary_domains_rejected(self):
        for choices in (["1", "2"], ["0", "1-3", "3-5"], ["0", "1", "1.0"],
                        ["1+", "2", "3"], ["0", "1", "2", "Other"],
                        ["0", "1", "2", "[generalised:unknown]"],
                        ["1. No", "2. Maybe", "3. Yes"], ["Not answered"]):
            self.assertIsNone(agent._ordered_coordinates(choices, "NA_GATED"), choices)

    def test_original_categorical_channels_and_missing_sentinels_preserved(self):
        frame, schema = self.fixture()
        frame.loc[0, "a"] = np.nan
        schema["items"]["a"]["gate"] = {"parent": "b"}
        frame.loc[1, "a"] = "NA_GATED"
        items, options, codes = agent._layout(frame, schema)
        features = agent._features(["a", "b"], options, codes, schema)
        self.assertEqual(list(features), ["g0", "g1", "o0"])
        self.assertEqual(features["g0"].iloc[:3].tolist(), ["missing", "v6", "v2"])
        self.assertTrue(features["o0"].iloc[[0, 1, 5]].isna().all())
        self.assertEqual(features["o0"].iloc[2], .5)
        self.assertEqual(features["g1"].iloc[:2].tolist(), ["v0", "v1"])

    def test_parameters_prior_expansion_and_payload_dependency_contract_unchanged(self):
        original = ast.parse((ROOT / "participant_r11/main.py").read_text())
        candidate = ast.parse((ROOT / "participant_r17/main.py").read_text())
        def selected(tree):
            return {getattr(n, "name", None) or n.targets[0].id: ast.dump(n)
                for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("_layout", "_prior", "_expand"))
                or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ("PARAMETERS", "MIN_LABELS", "PSEUDOCOUNT", "RESERVED_COLUMNS"))}
        self.assertEqual(selected(original), selected(candidate))
        for name in ("requirements.txt", "LICENSE"):
            self.assertEqual((ROOT / "participant_r11" / name).read_bytes(),
                             (ROOT / "participant_r17" / name).read_bytes())

    def test_all_visible_target_rows_mixed_routing_and_input_immutability(self):
        frame, schema = self.fixture()
        frame["q"] = [0, 1] * 48 + [np.nan] * 150
        schema["items"]["q"] = {"class": "PREDICT", "values": [0, 1]}
        original, original_schema = frame.copy(deep=True), copy.deepcopy(schema)
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            vectors = agent.predict(frame, schema)
        self.assertEqual([len(f["labels"]) for f in RecordingBackend.fits], [240, 96])
        for fit in RecordingBackend.fits:
            self.assertEqual(list(fit["frame"]), ["g0", "g1", "o0"])
            self.assertEqual(fit["cat_features"], [0, 1])
            self.assertEqual(fit["parameters"], agent.PARAMETERS)
        self.assertEqual(len(vectors), 156)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(schema, original_schema)

    def test_no_query_statistics_identifiers_roles_or_index_feature(self):
        frame, schema = self.fixture()
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            first = agent.predict(frame, schema)
        first_fit = RecordingBackend.fits[0]["frame"]
        extended = pd.concat([frame, frame.tail(6)], ignore_index=True)
        extended["respondent_id"], extended["role"] = 999999, "IGNORED"
        extended.index = np.arange(9000, 9000 + len(extended))
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            second = agent.predict(extended, schema)
        second_fit = RecordingBackend.fits[1]["frame"]
        pd.testing.assert_frame_equal(first_fit.reset_index(drop=True), second_fit.reset_index(drop=True))
        self.assertEqual(first, second[:6])
        self.assertEqual(first, second[6:])

    def test_canonical_missing_given_and_positive_unseen_gated_target_support(self):
        frame, schema = self.fixture()
        frame.loc[240, "a"] = np.nan
        schema["items"]["p"]["gate"] = {"parent": "b"}
        schema["items"]["p"]["observed_if"] = "b == impossible"
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            vectors = agent.predict(frame, schema)
        self.assertEqual(len(vectors), 7)
        self.assertEqual(len(vectors[0]), 6)  # first missing a, then p, at row240
        self.assertTrue(all(len(v) == 4 for v in vectors[1:]))
        for vector in vectors:
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1.)

    def test_unrecognized_given_retains_complete_original_feature_path(self):
        frame, schema = self.fixture()
        schema["items"]["a"]["values"] = ["prefix" + str(i) for i in range(6)]
        frame["a"] = ["prefix" + str(i) for i in range(6)] * 41
        with patch.object(agent, "CatBoostClassifier", RecordingBackend):
            agent.predict(frame, schema)
        self.assertEqual(list(RecordingBackend.fits[0]["frame"]), ["g0", "g1"])
        self.assertEqual(RecordingBackend.fits[0]["cat_features"], [0, 1])
        self.assertEqual(len(RecordingBackend.fits[0]["labels"]), 240)

    def test_real_pinned_mixed_numeric_nan_backend_learns_owned_control(self):
        frame, schema = self.fixture()
        frame.loc[[0, 5, 240, 245], "a"] = np.nan
        vectors = agent.predict(frame, schema)
        # Two missing GIVEN answers add canonical marginal vectors before p.
        target_vectors = [vectors[i] for i in (1, 2, 3, 4, 5, 7)]
        for vector, truth in zip(target_vectors[1:5], [0, 0, 1, 1]):
            self.assertGreater(vector[truth], .7)
        for vector in vectors:
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1.)

    def test_bad_visible_support_and_reserved_schema_still_fail_closed(self):
        frame, schema = self.fixture()
        frame.loc[0, "a"] = "999+"
        with self.assertRaisesRegex(ValueError, "outside schema"):
            agent.predict(frame, schema)
        frame, schema = self.fixture()
        schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaisesRegex(ValueError, "identifier"):
            agent.predict(frame, schema)

    def test_actual_three_public_schemas_coverage_without_questionname_selection(self):
        reference = ROOT / "_reference/data"
        counts = []
        for instrument in ("unhcr", "unicef", "world_bank"):
            schema = json.loads((reference / (instrument + ".json")).read_text())
            accepted = [agent._ordered_coordinates(item["values"], schema.get("gated_value"))
                        for item in schema["items"].values() if item["class"] == "GIVEN"]
            counts.append(sum(value is not None for value in accepted))
        self.assertEqual(counts, [9, 1, 3])


if __name__ == "__main__":
    unittest.main()
