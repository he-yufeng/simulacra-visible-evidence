"""Ten new native/backend boundary controls; not questionnaire performance."""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extratrees_r19", ROOT/"participant_r19/main.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class RecordingBackend:
    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, matrix, labels):
        self.classes_ = np.unique(labels)
        self.fits.append({"matrix": matrix.copy(), "labels": np.array(labels), "parameters": self.parameters})
        return self

    def predict_proba(self, matrix):
        return np.full((len(matrix),len(self.classes_)),1/len(self.classes_))


class ExtraTreeControls(unittest.TestCase):
    def fixture(self):
        pairs = np.tile(np.array([[0,0],[0,1],[1,0],[1,1]]), (40,1))
        labels = np.bitwise_xor(pairs[:,0],pairs[:,1])
        frame = pd.DataFrame({"a":pairs[:,0].tolist()+[0,0,1,1],
                              "b":pairs[:,1].tolist()+[0,1,0,1],
                              "p":labels.tolist()+[np.nan]*4,
                              "respondent_id":np.arange(164), "role":["TRAIN"]*160+["DEV"]*4})
        schema = {"gated_value":"NA_GATED", "items":{
            "a":{"class":"GIVEN","values":[0,1]},
            "b":{"class":"GIVEN","values":[0,1]},
            "p":{"class":"PREDICT","values":[0,1]},
            "respondent_id":{"class":"EXCLUDE","values":[]}}}
        return frame,schema

    def test_actual_backend_xor_and_positive_support(self):
        frame,schema = self.fixture()
        before = frame.copy(deep=True)
        vectors = agent.predict(frame,schema)
        self.assertEqual(len(vectors),4)
        for vector,truth in zip(vectors,[0,1,1,0]):
            self.assertGreater(vector[truth],.75)
            self.assertTrue(np.isfinite(vector).all())
            self.assertTrue((np.asarray(vector)>0).all())
            self.assertAlmostEqual(sum(vector),1)
        pd.testing.assert_frame_equal(frame,before)

    def test_native_query_multiplicity_identifiers_and_index_invariance(self):
        frame,schema = self.fixture()
        first = agent.predict(frame,schema)
        extended = pd.concat([frame,frame.tail(4)],ignore_index=True)
        extended["respondent_id"] = np.arange(8000,8000+len(extended))
        extended["role"] = "IGNORED"
        extended.index = np.arange(10000,10000+len(extended))
        second = agent.predict(extended,schema)
        np.testing.assert_allclose(first,second[:4],rtol=0,atol=1e-12)
        np.testing.assert_allclose(first,second[4:],rtol=0,atol=1e-12)

    def test_full_given_onehot_target_specific_visible_rows(self):
        frame,schema = self.fixture()
        frame["q"] = [0,1]*48+[np.nan]*68
        schema["items"]["q"] = {"class":"PREDICT","values":[0,1]}
        RecordingBackend.fits = []
        with patch.object(agent,"ExtraTreesClassifier",RecordingBackend):
            agent.predict(frame,schema)
        self.assertEqual([len(item["labels"]) for item in RecordingBackend.fits],[160,96])
        for item in RecordingBackend.fits:
            self.assertEqual(item["matrix"].shape[1],6)
            self.assertEqual(item["matrix"].dtype,np.float32)
            self.assertEqual(set(np.unique(item["matrix"])),{0,1})
            np.testing.assert_array_equal(item["matrix"].sum(axis=1),np.full(len(item["labels"]),2))
            self.assertEqual(item["parameters"],agent.PARAMETERS)
            self.assertEqual(item["parameters"]["n_jobs"],2)

    def test_schema_nominal_channels_with_explicit_missing(self):
        codes = {"a":np.array([1,0,-1])}
        matrix,offsets = agent._features(["a"],{"a":[20,10]},codes,3)
        np.testing.assert_array_equal(matrix,[[0,1,0],[1,0,0],[0,0,1]])
        self.assertEqual(offsets,{"a":(0,3)})

    def test_reverse_classes_keep_unseen_option_support(self):
        frame = pd.DataFrame({"a":[0,1]*50+[0], "p":[0,2]*50+[np.nan]})
        schema = {"gated_value":"NA_GATED", "items":{
            "a":{"class":"GIVEN","values":[0,1]}, "p":{"class":"PREDICT","values":[0,1,2]}}}
        class ReverseBackend(RecordingBackend):
            def fit(self,x,y):
                self.classes_ = np.array([2,0]);return self
            def predict_proba(self,x):
                return np.tile([.9,.1],(len(x),1))
        with patch.object(agent,"ExtraTreesClassifier",ReverseBackend):
            vector = agent.predict(frame,schema)[0]
        np.testing.assert_allclose(vector,np.array([10.5,.5,90.5])/101.5,rtol=0,atol=1e-15)

    def test_sparse_fallback_and_canonical_missing_order(self):
        frame = pd.DataFrame({"a":[0,1,0,np.nan,0,np.nan],"p":[0,1,0,1,np.nan,np.nan]})
        schema = {"gated_value":"NA_GATED", "items":{
            "a":{"class":"GIVEN","values":[0,1]},"p":{"class":"PREDICT","values":[0,1]}}}
        with patch.object(agent,"ExtraTreesClassifier",side_effect=AssertionError("No sparse fit")):
            self.assertEqual(agent.predict(frame,schema),[[.625,.375],[.5,.5],[.625,.375],[.5,.5]])

    def test_gate_remains_learned_supported_category(self):
        frame,schema = self.fixture()
        frame["p"] = frame["p"].astype(object)
        frame.loc[np.arange(0,160,7),"p"] = "NA_GATED"
        schema["items"]["p"]["gate"] = {"parent":"a"}
        schema["items"]["p"]["observed_if"] = "a == 1"
        with patch.object(agent,"ExtraTreesClassifier",RecordingBackend):
            vectors = agent.predict(frame,schema)
        for vector in vectors:
            self.assertEqual(len(vector),3)
            self.assertTrue((np.asarray(vector)>0).all())
            self.assertAlmostEqual(sum(vector),1)

    def test_empty_complete_single_class_constant_no_given(self):
        frame,schema = self.fixture()
        with patch.object(agent,"ExtraTreesClassifier",side_effect=AssertionError("No useful fit")):
            self.assertEqual(agent.predict(frame.tail(0),schema),[])
            self.assertEqual(agent.predict(frame.tail(4),schema),[[.5,.5]]*4)
            self.assertEqual(agent.predict(frame.iloc[:160],schema),[])
            mono = frame.copy(deep=True);mono.loc[:159,"p"] = 0
            np.testing.assert_allclose(agent.predict(mono,schema),np.tile([160.5/161,.5/161],(4,1)),atol=1e-15)
            constant = frame.copy(deep=True);constant.loc[:159,["a","b"]] = 0
            self.assertEqual(agent.predict(constant,schema),[[.5,.5]]*4)
            self.assertEqual(agent.predict(frame[["p"]],{"gated_value":"NA_GATED","items":{"p":schema["items"]["p"]}}),[[.5,.5]]*4)

    def test_invalid_input_class_map_and_probabilities_fail(self):
        frame,schema = self.fixture();frame.loc[0,"p"] = 9
        with self.assertRaisesRegex(ValueError,"outside schema"):
            agent.predict(frame,schema)
        frame,schema = self.fixture();schema["items"]["respondent_id"]["class"] = "GIVEN"
        with self.assertRaisesRegex(ValueError,"identifier"):
            agent.predict(frame,schema)
        for classes in ([0,0],[0,3],[0,1.5]):
            with self.assertRaisesRegex(ValueError,"support"):
                agent._expand([[.5,.5]],classes,3,100)
        for values in ([[np.nan,.5]],[[-.1,1.1]],[[.2,.2]]):
            with self.assertRaisesRegex(ValueError,"probabilities"):
                agent._expand(values,[0,1],2,100)

    def test_unchanged_owned_boundaries_and_no_external_io(self):
        old = ast.parse((ROOT/"participant_r11/main.py").read_text())
        new = ast.parse((ROOT/"participant_r19/main.py").read_text())
        for name in ("_layout","_prior","_expand"):
            a = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b = next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        imports = {n.module for n in ast.walk(new) if isinstance(n,ast.ImportFrom)}
        imports |= {a.name for n in ast.walk(new) if isinstance(n,ast.Import) for a in n.names}
        self.assertEqual(imports,{"numpy","pandas","sklearn.ensemble"})
        self.assertEqual(agent.MIN_LABELS,90);self.assertEqual(agent.PSEUDOCOUNT,.5)
        self.assertEqual(agent.PARAMETERS["random_state"],20261020)


if __name__ == "__main__":
    unittest.main()
