"""Owned selector/whole-block fixtures with fake experts, not accuracy evidence."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "participant_r09"
sys.path.insert(0, str(FOLDER))
try:
    spec = importlib.util.spec_from_file_location("mixture_r09", FOLDER / "main.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
finally:
    sys.path.pop(0)


class WholeBlockMixtureTests(unittest.TestCase):
    def fixture(self, repeats=1):
        z = np.tile([0, 1], 120)
        frame = pd.DataFrame({"given": z.tolist() + [0, 1] * repeats,
            "p": z.tolist() + [np.nan] * (2 * repeats),
            "q": z.tolist() + [np.nan] * (2 * repeats),
            "respondent_id": np.arange(240 + 2 * repeats)})
        schema = {"gated_value": "NA_GATED", "items": {
            "given": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]},
            "q": {"class": "PREDICT", "values": [0, 1], "gate": {"parent": "given"}},
            "respondent_id": {"class": "EXCLUDE", "values": []}}}
        return frame, schema

    def expert(self, frame, schema, *, latent=False, helpful=True):
        output = []
        for row in range(len(frame)):
            truth = int(frame["given"].iloc[row])
            for name in ("p", "q"):
                if not pd.isna(frame[name].iloc[row]):
                    continue
                correct = (.9 if helpful else .2) if latent else .6
                if name == "q":
                    correct = .2 if latent else .85
                vector = [1 - correct, 1 - correct]
                vector[truth] = correct
                vector[1 - truth] = 1 - correct
                if name == "q":
                    vector[1 - truth] -= .05
                    vector += [.05]
                output.append(vector)
        return output

    def mocks(self, helpful=True):
        return (patch.object(agent.evidence_core, "predict", side_effect=lambda f, s: self.expert(f, s)),
                patch.object(agent.latent_core, "predict", side_effect=lambda f, s:
                             self.expert(f, s, latent=True, helpful=helpful)))

    def test_both_experts_receive_entire_validation_predict_block_masked(self):
        frame, schema = self.fixture()
        before = frame.copy(deep=True)
        masks = self.mocks()
        with masks[0] as base, masks[1] as latent:
            agent.predict(frame, schema)
        for calls in (base.call_args_list[:3], latent.call_args_list[:3]):
            self.assertEqual(len(calls), 3)
            for fold, call in enumerate(calls):
                seen = call.args[0]
                valid = np.flatnonzero(np.arange(240) % 3 == fold)
                train = np.flatnonzero(np.arange(240) % 3 != fold)
                self.assertTrue(seen.iloc[valid][["p", "q"]].isna().all().all())
                self.assertTrue(seen.iloc[train][["p", "q"]].notna().all().all())
                pd.testing.assert_series_equal(seen["given"], before["given"])
        pd.testing.assert_frame_equal(frame, before)

    def test_half_mixture_selected_per_target_with_exact_other_target_fallback(self):
        frame, schema = self.fixture()
        masks = self.mocks()
        with masks[0], masks[1]:
            actual = agent.predict(frame, schema)
        np.testing.assert_allclose(actual[0], [.75, .25])
        np.testing.assert_allclose(actual[2], [.25, .75])
        expected = self.expert(frame, schema)
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(actual[3], expected[3])
        for vector in actual:
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.array(vector) >= 0).all())
            self.assertAlmostEqual(sum(vector), 1)

    def test_no_oof_gain_returns_exact_baseline_and_does_not_refit_latent(self):
        frame, schema = self.fixture()
        masks = self.mocks(helpful=False)
        with masks[0] as base, masks[1] as latent:
            actual = agent.predict(frame, schema)
        self.assertEqual(actual, self.expert(frame, schema))
        self.assertEqual(base.call_count, 4)
        self.assertEqual(latent.call_count, 3)

    def test_extra_fully_hidden_queries_do_not_change_selection_or_predictions(self):
        masks = self.mocks()
        with masks[0], masks[1]:
            one = agent.predict(*self.fixture(repeats=1))
            many = agent.predict(*self.fixture(repeats=3))
        self.assertEqual(many, one * 3)

    def test_missing_training_truth_is_not_a_label_and_insufficient_target_stays_base(self):
        frame, schema = self.fixture()
        frame.loc[60:239, "q"] = np.nan
        masks = self.mocks()
        with masks[0], masks[1]:
            items, options, codes = agent._layout(frame, schema)
            selected = agent._select_mixtures(frame, schema, items, options, codes)
        self.assertEqual(selected, {"p"})

    def test_small_sample_runs_actual_frozen_base_without_any_meta_or_latent_calls(self):
        frame, schema = self.fixture()
        frame = pd.concat([frame.iloc[:4], frame.iloc[-2:]], ignore_index=True)
        original = agent.evidence_core.predict(frame, schema)
        with patch.object(agent.latent_core, "predict", side_effect=AssertionError("not needed")), \
                patch.object(agent._select_mixtures.__globals__["evidence_core"], "predict",
                             wraps=agent.evidence_core.predict) as base:
            self.assertEqual(agent.predict(frame, schema), original)
        self.assertEqual(base.call_count, 1)

    def test_invalid_expert_output_or_undeclared_answer_fails_closed(self):
        frame, schema = self.fixture()
        with patch.object(agent.evidence_core, "predict", return_value=[]):
            with self.assertRaisesRegex(ValueError, "canonical missing cells"):
                agent.predict(frame, schema)
        with self.assertRaisesRegex(ValueError, "invalid expert probability"):
            agent._vector([np.nan, 1], 2)
        frame.iloc[0, frame.columns.get_loc("given")] = 99
        with self.assertRaisesRegex(ValueError, "outside declared option support"):
            agent.predict(frame, schema)

    def test_frozen_core_bytes_and_identifier_index_invariance(self):
        for name, expected in (("evidence_core.py", "86d0eb535267f338e05e31aae8585198f953202941e1a7cc03fddccfcffc1184"),
                               ("latent_core.py", "04680d52dacf336d2f7c1d90925ff754ab9176425a1389d727f3c61c5f2350c7")):
            self.assertEqual(hashlib.sha256((FOLDER / name).read_bytes()).hexdigest(), expected)
        frame, schema = self.fixture()
        masks = self.mocks()
        with masks[0], masks[1]:
            first = agent.predict(frame, schema)
            frame["respondent_id"] += 7654321
            frame.index = np.arange(len(frame)) * 7 + 13
            self.assertEqual(first, agent.predict(frame, schema))


if __name__ == "__main__":
    unittest.main()
