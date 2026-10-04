"""New observation-driver checks; no old model tests or accuracy claim."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from observe_r09_selection import always_half, capture


class ObserverTests(unittest.TestCase):
    def fixture(self, selected=True):
        frame, schema = object(), object()
        original_calls = []

        def expert(name, output):
            def run(f, s):
                original_calls.append((name, f))
                return output
            return run

        evidence = SimpleNamespace(predict=expert("r05", [[.8, .2]]))
        latent = SimpleNamespace(predict=expert("latent", [[.2, .8]]))
        agent = SimpleNamespace(evidence_core=evidence, latent_core=latent)

        def selector(f, s):
            for _ in range(3):
                masked = object()
                evidence.predict(masked, s)
                latent.predict(masked, s)
            return {"p"} if selected else set()

        agent._select_mixtures = selector

        def predict(f, s):
            chosen = agent._select_mixtures(f, s)
            base = evidence.predict(f, s)
            if not chosen:
                return base
            alternative = latent.predict(f, s)
            return always_half(base, alternative, ["p"], {"p"})

        agent.predict = predict
        return agent, frame, schema, original_calls

    def test_actual_calls_not_replaced_by_full_fit_cache(self):
        agent, frame, schema, calls = self.fixture()
        result = capture(agent, frame, schema)
        self.assertEqual(len(calls), 8)
        self.assertEqual(sum(f is frame for _, f in calls), 2)
        self.assertEqual(result["selected"], {"p"})
        self.assertEqual(result["r09"], [[.5, .5]])
        self.assertEqual(result["r05"], [[.8, .2]])
        self.assertFalse(result["extra_full_latent_for_ablation"])

    def test_skipped_latent_is_explicit_extra_control_call(self):
        agent, frame, schema, calls = self.fixture(False)
        result = capture(agent, frame, schema)
        self.assertIs(result["r09"], result["r05"])
        self.assertEqual(result["selected"], set())
        self.assertTrue(result["extra_full_latent_for_ablation"])
        self.assertTrue(result["calls"][-1]["ablation_only"])
        self.assertEqual(len(calls), 8)

    def test_original_functions_restored_after_capture(self):
        agent, frame, schema, _ = self.fixture()
        before = (agent.evidence_core.predict, agent.latent_core.predict, agent._select_mixtures)
        capture(agent, frame, schema)
        self.assertEqual(before, (agent.evidence_core.predict, agent.latent_core.predict,
                                  agent._select_mixtures))

    def test_control_blends_every_predict_target_and_keeps_given(self):
        base = [[.8, .2], [.9, .1], [.7, .3]]
        alternative = [[.2, .8], [.1, .9], [.3, .7]]
        result = always_half(base, alternative, ["p", "q", "g"], {"p", "q"})
        self.assertEqual(result[:2], [[.5, .5], [.5, .5]])
        self.assertIs(result[2], base[2])

    def test_noncanonical_control_counts_and_widths_fail(self):
        with self.assertRaises(ValueError):
            always_half([[.5, .5]], [], ["p"], {"p"})
        with self.assertRaises(ValueError):
            always_half([[.5, .5]], [[1]], ["p"], {"p"})


if __name__ == "__main__":
    unittest.main()
