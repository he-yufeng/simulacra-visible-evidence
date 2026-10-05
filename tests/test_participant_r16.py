"""Thirteen new linear-residual controls; never substitute for full survey scores."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("network_r16", ROOT / "participant_r16/main.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class LinearResidualTests(unittest.TestCase):
    def fixture(self, n=120, queries=4):
        z = np.arange(n) % 2
        frame = pd.DataFrame({"given": z.tolist() + (np.arange(queries) % 2).tolist(),
            "p": pd.Series(z.tolist() + [np.nan] * queries, dtype=object),
            "q": pd.Series(([0, 1, "NA_GATED"] * ((n + 2) // 3))[:n] + [np.nan] * queries, dtype=object),
            "respondent_id": np.arange(n + queries), "role": ["TRAIN"] * n + ["DEV"] * queries})
        schema = {"gated_value": "NA_GATED", "items": {
            "given": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]},
            "q": {"class": "PREDICT", "values": [0, 1], "gate": {"parent": "given"}},
            "respondent_id": {"class": "EXCLUDE", "values": []}}}
        return frame, schema

    def fake_fit(self, design, labels, widths, priors):
        return {"W0": np.zeros((design.shape[1], sum(widths.values()))),
            "W1": np.zeros((design.shape[1], agent.HIDDEN)), "b1": np.zeros(agent.HIDDEN),
            "W2": np.zeros((agent.HIDDEN, sum(widths.values()))),
            "b2": np.concatenate([np.log(priors[name]) for name in widths])}

    def tiny(self):
        rng = np.random.default_rng(202610516)
        x = rng.normal(0, .3, (4, 3))
        parameters = {"W0": rng.normal(0, .1, (3, 5)),
            "W1": rng.normal(0, .2, (3, 4)), "b1": rng.normal(0, .1, 4),
            "W2": rng.normal(0, .2, (4, 5)), "b2": rng.normal(0, .1, 5)}
        labels = {"p": np.array([0, 1, -1, 1]), "q": np.array([2, -1, 0, 1])}
        return x, labels, {"p": 2, "q": 3}, parameters

    def test_grouped_log_softmax_full_unequal_support_and_extreme_stability(self):
        logits = np.array([[1., 2., -3., 4., 5.], [-1000, 1000, -2000, 0, 2000]])
        logs = agent._log_softmax(logits, [2, 3])
        self.assertTrue(np.isfinite(logs).all())
        probabilities = np.exp(logs)
        np.testing.assert_allclose(probabilities[:, :2].sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[:, 2:].sum(axis=1), 1)
        expected = logits[0, :2] - np.log(np.exp(logits[0, :2]).sum())
        np.testing.assert_allclose(logs[0, :2], expected)
        for widths in ([], [0, 5], [2, 2]):
            with self.assertRaises(ValueError):
                agent._log_softmax(logits, widths)
        with self.assertRaises(ValueError):
            agent._log_softmax(np.array([[np.nan, 0]]), [2])

    def test_all_parameter_gradients_match_central_finite_differences(self):
        x, labels, widths, parameters = self.tiny()
        _, gradient = agent._loss_gradient(x, labels, widths, parameters)
        for name, array in parameters.items():
            for index in np.ndindex(array.shape):
                before = float(array[index])
                array[index] = before + 1e-6
                plus = agent._loss_gradient(x, labels, widths, parameters)[0]
                array[index] = before - 1e-6
                minus = agent._loss_gradient(x, labels, widths, parameters)[0]
                array[index] = before
                self.assertAlmostEqual((plus - minus) / 2e-6, gradient[name][index], places=7)

    def test_fully_hidden_rows_have_zero_label_loss_or_gradient_effect(self):
        x, labels, widths, parameters = self.tiny()
        labels["q"][2] = -1
        before, gradient = agent._loss_gradient(x, labels, widths, parameters)
        altered = x.copy()
        altered[2] = [5, 7, 9]
        after, changed = agent._loss_gradient(altered, labels, widths, parameters)
        self.assertAlmostEqual(before, after, places=13)
        for name in gradient:
            np.testing.assert_allclose(gradient[name], changed[name], rtol=0, atol=1e-13)
        labels["p"][0] = 5
        with self.assertRaises(ValueError):
            agent._loss_gradient(x, labels, widths, parameters)

    def test_halfcount_prior_keeps_unseen_and_gate_options_positive(self):
        prior = agent._prior(np.array([0, 0, 1, -1]), 3)
        np.testing.assert_allclose(prior, np.array([2.5, 1.5, .5]) / 4.5)
        self.assertTrue((prior > 0).all())
        np.testing.assert_allclose(agent._prior(np.array([-1, -1]), 4), np.full(4, .25))
        extreme = np.exp(agent._log_softmax(np.array([[2000., -2000., 0.]]), [3]))
        mixed = .95 * extreme + .05 * prior
        agent._checked(mixed, 3)

    def test_given_design_and_center_ignore_hidden_queries_and_ids(self):
        frame, schema = self.fixture(queries=2)
        extended, _ = self.fixture(queries=40)
        extended.loc[120:, "given"] = 1
        extended["respondent_id"] += 900000
        extended["role"] = "IGNORED"
        extended.index = np.arange(10000, 10000 + len(extended))
        _, options, codes = agent._layout(frame, schema)
        _, extra_options, extra_codes = agent._layout(extended, schema)
        rows = np.arange(120)
        design = agent._design(codes, options, ["given"], rows)
        extra = agent._design(extra_codes, extra_options, ["given"], rows)
        np.testing.assert_array_equal(design[:120], extra[:120])
        np.testing.assert_allclose(design[:120].mean(axis=0), 0, atol=1e-15)
        with patch.object(agent, "_fit", side_effect=self.fake_fit):
            first = agent.predict(frame, schema)
            many = agent.predict(extended, schema)
        self.assertEqual(many[:4], first)

    def test_one_shared_fit_receives_visible_rows_and_only_given_design(self):
        frame, schema = self.fixture(n=180, queries=20)
        frame.loc[0:29, "q"] = np.nan
        with patch.object(agent, "_fit", side_effect=self.fake_fit) as spy:
            result = agent.predict(frame, schema)
        self.assertEqual(spy.call_count, 1)
        design, labels, widths, _ = spy.call_args.args
        self.assertEqual(design.shape, (180, 3))
        self.assertEqual(set(widths), {"p", "q"})
        self.assertEqual(len(labels["p"]), 180)
        self.assertEqual(int((labels["q"] < 0).sum()), 30)
        self.assertEqual(len(result), 70)

    def test_no_given_small_labels_single_class_and_constant_features_use_priors(self):
        for mode in ("no_given", "small", "single", "constant"):
            frame, schema = self.fixture(n=89 if mode == "small" else 120)
            if mode == "no_given":
                schema["items"]["given"]["class"] = "EXCLUDE"
            elif mode == "single":
                frame.loc[:119, "p"] = 0
                frame.loc[:119, "q"] = 0
            elif mode == "constant":
                frame.loc[:119, "given"] = 0
            with patch.object(agent, "_fit", side_effect=AssertionError("must not fit")):
                vectors = agent.predict(frame, schema)
            self.assertEqual(len(vectors), 8)
            self.assertTrue(all(np.isfinite(v).all() and (np.asarray(v) > 0).all() for v in vectors))

    def test_full_canonical_queries_and_input_schema_are_unchanged(self):
        frame, schema = self.fixture(queries=2)
        frame.loc[0, "given"] = np.nan
        frame.loc[10, "p"] = np.nan
        before, frozen = frame.copy(deep=True), copy.deepcopy(schema)
        with patch.object(agent, "_fit", side_effect=self.fake_fit):
            vectors = agent.predict(frame, schema)
        self.assertEqual(len(vectors), 6)
        self.assertEqual([len(v) for v in vectors], [2, 2, 2, 3, 2, 3])
        for vector in vectors:
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector) > 0).all())
            self.assertAlmostEqual(sum(vector), 1)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(schema, frozen)

    def test_invalid_visible_values_reserved_identifiers_and_empty_support_refused(self):
        frame, schema = self.fixture()
        invalid = frame.copy()
        invalid.loc[123, "given"] = 99
        with self.assertRaises(ValueError):
            agent.predict(invalid, schema)
        invalid_schema = copy.deepcopy(schema)
        invalid_schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaises(ValueError):
            agent.predict(frame, invalid_schema)
        invalid_schema = copy.deepcopy(schema)
        invalid_schema["items"]["p"]["values"] = []
        with self.assertRaises(ValueError):
            agent.predict(frame, invalid_schema)

    def test_illegal_network_outputs_and_memory_stop_do_not_silently_fallback(self):
        for bad in ([[np.nan, .5]], [[0, 1]], [[-.1, 1.1]], [[.2, .2]], [[.2, .3, .5]]):
            with self.assertRaises(ValueError):
                agent._checked(bad, 2)
        frame, schema = self.fixture()
        def broken(*args):
            parameters = self.fake_fit(*args)
            parameters["W2"][0, 0] = np.nan
            return parameters
        with patch.object(agent, "_fit", side_effect=broken):
            with self.assertRaises(ValueError):
                agent.predict(frame, schema)
        with patch.object(agent, "MAX_WORKSPACE_BYTES", 1), patch.object(agent, "_fit") as spy:
            with self.assertRaisesRegex(ValueError, "no thinning"):
                agent.predict(frame, schema)
        self.assertEqual(spy.call_count, 0)

    def test_fixed_seed_optimizer_is_deterministic_on_new_owned_tiny_data(self):
        x = np.column_stack((np.linspace(-1, 1, 96), np.ones(96)))
        labels = {"p": (x[:, 0] > 0).astype(np.int64)}
        widths, priors = {"p": 2}, {"p": np.array([.5, .5])}
        first = agent._fit(x, labels, widths, priors)
        second = agent._fit(x, labels, widths, priors)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_actual_new_multitarget_network_one_fit_learns_xor_and_full_support(self):
        x = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (160, 1))
        queries = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        frame = pd.DataFrame({"a": x[:, 0].tolist() + queries[:, 0].tolist(),
            "b": x[:, 1].tolist() + queries[:, 1].tolist(),
            "p": pd.Series(np.bitwise_xor(x[:, 0], x[:, 1]).tolist() + [np.nan] * 4, dtype=object),
            "q": pd.Series(((x[:, 0] + 2*x[:, 1]) % 3).tolist() + [np.nan] * 4, dtype=object)})
        schema = {"gated_value": "NA_GATED", "items": {
            "a": {"class": "GIVEN", "values": [0, 1]}, "b": {"class": "GIVEN", "values": [0, 1]},
            "p": {"class": "PREDICT", "values": [0, 1]},
            "q": {"class": "PREDICT", "values": [0, 1, 2], "gate": {"parent": "a"}}}}
        with patch.object(agent, "_fit", wraps=agent._fit) as spy:
            vectors = agent.predict(frame, schema)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(vectors), 8)
        for row, (p, q) in enumerate(zip([0, 1, 1, 0], [0, 2, 1, 0])):
            self.assertGreater(vectors[2*row][p], .8)
            self.assertGreater(vectors[2*row+1][q], .75)
            self.assertEqual(len(vectors[2*row+1]), 4)
            self.assertGreater(vectors[2*row+1][-1], 0)


    def test_linear_carrier_retains_independent_logits_with_hidden_residual_zero(self):
        x = np.eye(3)
        carrier = np.array([[1., -1., 0., 0.], [0., 0., 2., -2.], [3., -3., 4., -4.]])
        params = {"W0": carrier, "W1": np.zeros((3, 1)), "b1": np.zeros(1),
            "W2": np.zeros((1, 4)), "b2": np.zeros(4)}
        hidden, logp = agent._forward(x, params, {"p": 2, "q": 2})
        np.testing.assert_array_equal(hidden, np.zeros((3, 1)))
        np.testing.assert_allclose(logp, agent._log_softmax(x @ carrier, [2, 2]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
