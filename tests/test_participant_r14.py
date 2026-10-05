"""Twelve new runtime-equivalence controls, not questionnaire quality scores."""
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
FOLDER = ROOT / "participant_r14"
sys.path.insert(0, str(FOLDER))
try:
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    agent = load("runtime_r14", FOLDER / "main.py")
    original = load("frozen_calibration_r13", ROOT / "participant_r13/main.py")
finally:
    sys.path.pop(0)


class RuntimeRepairTests(unittest.TestCase):
    def fixture(self, n=300, queries=4):
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

    def backend(self, frame, schema, uniform=False):
        vectors = []
        for row in range(len(frame)):
            for name, item in schema["items"].items():
                if item["class"] not in ("GIVEN", "PREDICT") or pd.notna(frame[name].iloc[row]):
                    continue
                width = len(item["values"]) + bool(item.get("gate"))
                vector = np.full(width, 1 / width)
                if name == "p" and not uniform:
                    value = frame["given"].iloc[row]
                    truth = int(value) if pd.notna(value) else 0
                    vector = np.array([.35, .35])
                    vector[truth] = .65
                vectors.append(vector.tolist())
        return vectors

    def test_eligible_crop_whole_block_mask_and_full_original_final_pass(self):
        frame, schema = self.fixture(queries=30)
        frame = frame.iloc[np.ravel(np.column_stack((np.arange(30), np.arange(300, 330)))).tolist() + list(range(30, 300))]
        before, frozen_schema = frame.copy(deep=True), copy.deepcopy(schema)
        with patch.object(agent.boost_core, "predict", side_effect=self.backend) as spy:
            agent.predict(frame, schema)
        self.assertEqual(spy.call_count, 2)
        compact = spy.call_args_list[0].args[0]
        eligible = np.flatnonzero(frame[["p", "q"]].notna().any(axis=1).to_numpy())
        self.assertEqual(len(compact), 300)
        validation = np.flatnonzero(np.arange(300) % 10 >= 7)
        self.assertTrue(compact.iloc[validation][["p", "q"]].isna().all().all())
        pd.testing.assert_frame_equal(compact.drop(columns=["p", "q"]), before.iloc[eligible].drop(columns=["p", "q"]))
        self.assertIs(spy.call_args_list[1].args[0], frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(schema, frozen_schema)

    def test_original_needed_target_survives_all_known_compact_input(self):
        frame, schema = self.fixture(queries=20)
        with patch.object(agent.boost_core, "predict", side_effect=self.backend):
            selected = agent._select(frame, schema, *agent.boost_core._layout(frame, schema))
        self.assertEqual(selected, {"p": .5})

    def test_unneeded_target_stays_unselected_despite_validation_masking(self):
        frame, schema = self.fixture()
        frame["q"] = [0, 1, "NA_GATED", 0] * 76
        with patch.object(agent.boost_core, "predict", side_effect=self.backend):
            selected = agent._select(frame, schema, *agent.boost_core._layout(frame, schema))
        self.assertEqual(selected, {"p": .5})
        self.assertNotIn("q", selected)

    def test_fake_equivalence_partial_labels_sparse_given_and_row_reordering(self):
        for mode in ("plain", "partial", "sparse", "reordered"):
            frame, schema = self.fixture(n=330, queries=10)
            if mode == "partial":
                frame.loc[100:329, "q"] = np.nan
            elif mode == "sparse":
                frame.loc[[0, 20, 332], "given"] = np.nan
            elif mode == "reordered":
                frame = frame.sample(frac=1, random_state=2026105)
                frame.index = np.arange(len(frame))[::-1] + 12000
            with patch.object(agent.boost_core, "predict", side_effect=self.backend):
                expected = original.predict(frame, schema)
                actual = agent.predict(frame, schema)
            np.testing.assert_allclose(np.concatenate(actual), np.concatenate(expected), rtol=0, atol=1e-15)

    def test_hidden_query_count_identifiers_and_index_do_not_change_selection(self):
        frame, schema = self.fixture(queries=2)
        extended, _ = self.fixture(queries=10)
        extended["respondent_id"] += 1000000
        extended["role"] = "IGNORED"
        extended.index = np.arange(5000, 5000 + len(extended))
        with patch.object(agent.boost_core, "predict", side_effect=self.backend):
            first = agent.predict(frame, schema)
            extra = agent.predict(extended, schema)
        self.assertEqual(extra, first * 5)

    def test_identity_fallback_keeps_original_list_object(self):
        frame, schema = self.fixture()
        baseline = self.backend(frame, schema, uniform=True)
        def backend(f, s):
            return baseline if f is frame else self.backend(f, s, uniform=True)
        with patch.object(agent.boost_core, "predict", side_effect=backend):
            self.assertIs(agent.predict(frame, schema), baseline)

    def test_small_and_only_given_missing_have_one_full_pass(self):
        for n in (0, 80, 299):
            frame, schema = self.fixture(n=n)
            with patch.object(agent.boost_core, "predict", side_effect=self.backend) as spy:
                self.assertEqual(agent.predict(frame, schema), self.backend(frame, schema))
            self.assertEqual(spy.call_count, 1)
        frame, schema = self.fixture(queries=0)
        frame.loc[0, "given"] = np.nan
        with patch.object(agent.boost_core, "predict", side_effect=self.backend) as spy:
            self.assertEqual(agent.predict(frame, schema), self.backend(frame, schema))
        self.assertEqual(spy.call_count, 1)

    def test_batched_groups_are_canonical_and_unselected_objects_exact(self):
        frame, schema = self.fixture(n=0, queries=2)
        frame.loc[0, "given"] = np.nan
        frame["q"] = frame["q"].astype(object)
        frame.loc[0, "q"] = "NA_GATED"
        vectors = [[.4, .6], [.6, .4], [.3, .7], [.1, .2, .7]]
        groups = agent._groups(frame, schema, vectors)
        np.testing.assert_array_equal(groups["given"][0], [0])
        np.testing.assert_array_equal(groups["p"][0], [1, 2])
        np.testing.assert_array_equal(groups["p"][1], [0, 1])
        np.testing.assert_array_equal(groups["q"][0], [3])
        frame, schema = self.fixture()
        frame.loc[0, "given"] = np.nan
        baseline = self.backend(frame, schema)
        def backend(f, s):
            return baseline if f is frame else self.backend(f, s)
        with patch.object(agent.boost_core, "predict", side_effect=backend):
            result = agent.predict(frame, schema)
        self.assertIs(result[0], baseline[0])
        self.assertIs(result[-1], baseline[-1])

    def test_bad_probabilities_widths_counts_and_unselected_targets_rejected(self):
        frame, schema = self.fixture(n=0, queries=1)
        vectors = self.backend(frame, schema)
        for bad in ([np.nan, .5], [0, 1], [-.1, 1.1], [.1, .1], [.2, .3, .5]):
            with self.assertRaises(ValueError):
                agent._groups(frame, schema, [bad, vectors[1]])
        for broken in ([], vectors[:1], vectors + [vectors[0]]):
            with self.assertRaises(ValueError):
                agent._groups(frame, schema, broken)
        frame, schema = self.fixture()
        baseline = self.backend(frame, schema)
        baseline[-1] = [0, .5, .5]
        def backend(f, s):
            return baseline if f is frame else self.backend(f, s)
        with patch.object(agent.boost_core, "predict", side_effect=backend):
            with self.assertRaises(ValueError):
                agent.predict(frame, schema)

    def test_unknown_values_reserved_features_and_frozen_parameters(self):
        frame, schema = self.fixture()
        frame.loc[303, "given"] = 99
        with self.assertRaises(ValueError):
            agent.predict(frame, schema)
        frame, schema = self.fixture()
        schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaises(ValueError):
            agent.predict(frame, schema)
        self.assertEqual(agent.boost_core.PARAMETERS["iterations"], 96)
        self.assertEqual(agent.TEMPERATURES, original.TEMPERATURES)
        for name in ("MIN_VISIBLE", "MIN_VALIDATION_LABELS", "MIN_LOG_GAIN", "SE_MULTIPLIER", "FLOOR"):
            self.assertEqual(getattr(agent, name), getattr(original, name))
        self.assertEqual(hashlib.sha256((FOLDER / "boost_core.py").read_bytes()).hexdigest(),
            "32f16501feeb7b6890262165d3f510b14b818208e0ee73aca716d847ead6cdf8")
        self.assertEqual(hashlib.sha256((ROOT / "participant_r13/main.py").read_bytes()).hexdigest(),
            "1e9d461273c0043bb2810ef20fe5354757a653c3912c25f98a747678e0108d83")

    def test_batched_scaling_equals_original_individual_complete_support(self):
        vectors = np.array([[.1, .2, .7], [.7, .2, .1], [.00001, .49999, .5]])
        for temperature in original.TEMPERATURES:
            expected = np.vstack([original._scaled(vector, temperature) for vector in vectors])
            np.testing.assert_allclose(agent._scaled(vectors, temperature), expected, rtol=0, atol=1e-15)

    def test_new_actual_backend_old_and_repaired_outputs_match_with_four96_fits(self):
        x = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (120, 1))
        labels = np.bitwise_xor(x[:, 0], x[:, 1]).astype(object)
        labels[np.arange(len(labels)) % 41 == 0] = "NA_GATED"
        queries = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (5, 1))
        frame = pd.DataFrame({"a": x[:, 0].tolist() + queries[:, 0].tolist(),
            "b": x[:, 1].tolist() + queries[:, 1].tolist(),
            "p": labels.tolist() + [np.nan] * len(queries)})
        frame.loc[483, "a"] = np.nan
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": [0, 1]}, "b": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1], "gate": {"parent": "a"}}}}
        real, models = agent.boost_core.CatBoostClassifier, []
        def factory(**parameters):
            model = real(**parameters)
            models.append(model)
            return model
        with patch.object(agent.boost_core, "CatBoostClassifier", side_effect=factory):
            before = original.predict(frame, schema)
            after = agent.predict(frame, schema)
        self.assertEqual(len(models), 4)
        self.assertTrue(all(model.tree_count_ == 96 for model in models))
        self.assertEqual(len(before), 21)
        self.assertEqual(len(after), 21)
        np.testing.assert_allclose(np.concatenate(after), np.concatenate(before), rtol=0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
