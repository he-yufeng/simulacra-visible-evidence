"""Twelve inference/information-boundary controls, not private-data scores."""
import ast
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"participant_r20"))
import tree_core as core
spec = importlib.util.spec_from_file_location("chowliu_r20",ROOT/"participant_r20/main.py")
agent = importlib.util.module_from_spec(spec);spec.loader.exec_module(agent)


def hand_tree():
    return {"widths":[2,3,2],"parents":np.array([-1,0,1]),"order":[0,1,2],
            "children":[[1],[2],[]],"root_log_prior":np.log([.3,.7]),
            "log_conditionals":[None,np.log([[.7,.2,.1],[.1,.3,.6]]),np.log([[.8,.2],[.6,.4],[.1,.9]])]}


def enumerated(tree, evidence):
    values = [np.zeros(width) for width in tree["widths"]]
    total = 0.
    for state in itertools.product(*(range(width) for width in tree["widths"])):
        if any(observed>=0 and state[node]!=observed for node,observed in enumerate(evidence)):
            continue
        mass = float(np.exp(tree["root_log_prior"][state[0]]))
        for node in tree["order"][1:]:
            mass *= float(np.exp(tree["log_conditionals"][node][state[tree["parents"][node]],state[node]]))
        total += mass
        for node,answer in enumerate(state):
            values[node][answer] += mass
    return [value/total for value in values]


class DensityTreeControls(unittest.TestCase):
    def fixture(self):
        records = []
        for x in [0,1]:
            for middle in [0,1]:
                number = 400 if middle==x else 100
                for y in [0,1]:
                    count = number*9//10 if y==middle else number//10
                    records += [{"g":x,"m":middle,"y":y}]*count
        frame = pd.DataFrame(records+[{"g":0,"m":np.nan,"y":np.nan},{"g":1,"m":np.nan,"y":np.nan}])
        frame["respondent_id"] = np.arange(len(frame));frame["role"] = "IGNORED"
        schema = {"gated_value":"NA_GATED","items":{
            "g":{"class":"GIVEN","values":[0,1]},
            "m":{"class":"PREDICT","values":[0,1]},
            "y":{"class":"PREDICT","values":[0,1]},
            "respondent_id":{"class":"EXCLUDE","values":[]}}}
        return frame,schema

    def test_exact_inference_matches_independent_enumeration(self):
        tree = hand_tree()
        evidence = np.array([[-1,-1,-1],[0,-1,-1],[-1,2,-1],[-1,-1,1],[1,-1,0],[0,1,1]])
        result = core.infer_tree(tree,evidence)
        for row,observed in enumerate(evidence):
            expected = enumerated(tree,observed)
            for node,value in enumerate(expected):
                np.testing.assert_allclose(result[node][row],value,rtol=0,atol=1e-12)

    def test_batching_preserves_every_row_and_node(self):
        tree = hand_tree();evidence = np.tile(np.array([[-1,2,-1],[0,-1,1]]),(150,1))
        full = core.infer_tree(tree,evidence)
        with patch.object(core,"BATCH_ROWS",1):
            tiny = core.infer_tree(tree,evidence)
        for a,b in zip(full,tiny):
            self.assertEqual(len(a),300);np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

    def test_actual_chain_learning_positive_canonical_outputs(self):
        frame,schema = self.fixture();before = frame.copy(deep=True)
        values = agent.predict(frame,schema)
        self.assertEqual(len(values),4)
        for vector,truth in zip(values,[0,0,1,1]):
            self.assertGreater(vector[truth],.65)
            self.assertTrue(np.isfinite(vector).all());self.assertTrue((np.asarray(vector)>0).all())
            self.assertAlmostEqual(sum(vector),1)
        pd.testing.assert_frame_equal(frame,before)

    def test_explicit_pairwise_only_xor_limitation_not_hidden(self):
        pairs = np.tile(np.array([[0,0],[0,1],[1,0],[1,1]]),(100,1))
        labels = np.bitwise_xor(pairs[:,0],pairs[:,1])
        frame = pd.DataFrame({"a":pairs[:,0].tolist()+[0,0,1,1],"b":pairs[:,1].tolist()+[0,1,0,1],
                              "p":labels.tolist()+[np.nan]*4})
        schema = {"gated_value":"NA_GATED","items":{name:{"class":"GIVEN" if name!='p' else "PREDICT","values":[0,1]} for name in ['a','b','p']}}
        np.testing.assert_allclose(agent.predict(frame,schema),[[.5,.5]]*4,rtol=0,atol=1e-12)

    def test_hidden_query_count_index_and_role_invariance(self):
        frame,schema = self.fixture();first = agent.predict(frame,schema)
        extended = pd.concat([frame,frame.tail(2)],ignore_index=True)
        extended.index = np.arange(9000,9000+len(extended));extended['role'] = "CHANGED"
        extended['respondent_id'] = np.arange(5000,5000+len(extended))
        second = agent.predict(extended,schema)
        np.testing.assert_allclose(first,second[:4],rtol=0,atol=1e-12)
        np.testing.assert_allclose(first,second[4:],rtol=0,atol=1e-12)

    def test_fit_all_visible_targets_but_never_hidden_query_given(self):
        frame,schema = self.fixture();calls = []
        def record(data,widths):
            calls.append(data.copy());return core.fit_tree(data,widths)
        with patch.object(agent,"fit_tree",record):
            agent.predict(frame,schema)
            changed = frame.copy(deep=True);changed.loc[1000:,"g"] = [1,0]
            agent.predict(changed,schema)
        self.assertEqual(calls[0].shape,(1000,3))
        np.testing.assert_array_equal(calls[0],calls[1])
        self.assertTrue(np.isin(calls[0][:,1:],[0,1]).all())

    def test_partial_labels_joint_count_excludes_missing_not_zero(self):
        table,number = core.joint_counts(np.array([0,1,-1,0]),np.array([1,-1,0,0]),2,2)
        self.assertEqual(number,2);np.testing.assert_array_equal(table,[[1,1],[0,0]])
        frame = pd.DataFrame({"g":[0,1,0,1],"p":[0,np.nan,1,np.nan],"q":[np.nan,1,0,np.nan]})
        schema = {"gated_value":"NA_GATED","items":{name:{"class":"GIVEN" if name=='g' else "PREDICT","values":[0,1]} for name in ['g','p','q']}}
        before = frame.copy(deep=True);values = agent.predict(frame,schema)
        self.assertEqual(len(values),4)
        for value in values:
            self.assertAlmostEqual(sum(value),1);self.assertTrue((np.asarray(value)>0).all())
        pd.testing.assert_frame_equal(frame,before)

    def test_gated_sentinel_is_soft_supported_category(self):
        frame,schema = self.fixture();frame['y'] = frame['y'].astype(object)
        frame.loc[np.arange(0,1000,7),'y'] = "NA_GATED"
        schema['items']['y']['gate'] = {'parent':'m'};schema['items']['y']['observed_if'] = 'm == 1'
        values = agent.predict(frame,schema)
        for vector in [values[1],values[3]]:
            self.assertEqual(len(vector),3);self.assertTrue((np.asarray(vector)>0).all())
            self.assertAlmostEqual(sum(vector),1)

    def test_query_only_empty_complete_and_constant_single_node(self):
        frame,schema = self.fixture()
        self.assertEqual(agent.predict(frame.tail(0),schema),[])
        self.assertEqual(agent.predict(frame.iloc[:1000],schema),[])
        np.testing.assert_allclose(agent.predict(frame.tail(2),schema),[[.5,.5]]*4,atol=1e-12)
        single = pd.DataFrame({'p':[0,0,np.nan]})
        self.assertEqual(agent.predict(single,{'gated_value':'NA_GATED','items':{'p':{'class':'PREDICT','values':[0]}}}),[[1.]])

    def test_positive_extreme_likelihood_log_space(self):
        tree = {'widths':[2,2],'parents':np.array([-1,0]),'order':[0,1],'children':[[1],[]],
                'root_log_prior':np.log([1e-200,1-1e-200]),
                'log_conditionals':[None,np.log([[1-1e-200,1e-200],[1e-200,1-1e-200]])]}
        values = core.infer_tree(tree,np.array([[-1,0],[-1,-1]]))
        for vector in values[0]:
            self.assertTrue(np.isfinite(vector).all());self.assertTrue((vector>0).all());self.assertAlmostEqual(sum(vector),1)

    def test_outside_support_and_identifiers_refused(self):
        frame,schema = self.fixture();frame.loc[0,'y'] = 9
        with self.assertRaisesRegex(ValueError,'outside schema'):
            agent.predict(frame,schema)
        frame,schema = self.fixture();schema['items']['respondent_id']['class'] = 'GIVEN'
        with self.assertRaisesRegex(ValueError,'identifier'):
            agent.predict(frame,schema)
        with self.assertRaisesRegex(ValueError,'node code'):
            core.fit_tree(np.array([[0,2]]),[2,2])
        with self.assertRaisesRegex(ValueError,'support'):
            core.infer_tree(hand_tree(),np.array([[0,4,1]]))

    def test_full_schema_boundary_helpers_and_no_external_io(self):
        old = ast.parse((ROOT/'participant_r11/main.py').read_text());new = ast.parse((ROOT/'participant_r20/main.py').read_text())
        for name in ['_layout','_prior']:
            a = next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name==name)
            b = next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        tree = core.fit_tree(np.tile(np.array([[0,0,1],[1,1,0]]),(100,1)),[2,2,2])
        self.assertEqual(len(tree['order']),3);self.assertEqual(set(tree['order']),{0,1,2})
        for name in ['main.py','tree_core.py']:
            parsed = ast.parse((ROOT/'participant_r20'/name).read_text())
            calls = [node.func.id for node in ast.walk(parsed) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
            self.assertFalse({'open','eval','exec','print'}.intersection(calls))
        self.assertEqual(core.PSEUDOCOUNT,.5);self.assertEqual(core.MIN_JOINT,90)


if __name__ == '__main__':
    unittest.main()
