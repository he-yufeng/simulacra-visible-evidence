"""Positive categorical dependence-tree fit and log-space exact inference.

Owned implementation of established Chow-Liu-style estimation/sum-product.
Regularized available-case MI is not an optimal complete-data MLE guarantee.
No causal, high-order-interaction, or private-data generalization claim.
"""
import numpy as np
from scipy.special import logsumexp

PSEUDOCOUNT = .5
MIN_JOINT = 90
BATCH_ROWS = 128
UNDERFLOW_FLOOR = 1e-300


def joint_counts(left, right, width_left, width_right):
    valid = (left >= 0) & (right >= 0)
    ids = left[valid]*width_right+right[valid]
    return np.bincount(ids, minlength=width_left*width_right).reshape(width_left,width_right).astype(np.float64), int(valid.sum())


def fit_tree(data, widths):
    """Every node/pair, fitted data supplied by visible-only wrapper."""
    data = np.asarray(data,dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != len(widths) or not widths:
        raise ValueError("invalid categorical matrix")
    if any(width < 1 for width in widths):
        raise ValueError("empty node support")
    for index,width in enumerate(widths):
        if np.any((data[:,index] < -1) | (data[:,index] >= width)):
            raise ValueError("invalid node code")
    nodes = len(widths)
    priors = []
    for index,width in enumerate(widths):
        counts = np.bincount(data[data[:,index]>=0,index],minlength=width).astype(np.float64)+PSEUDOCOUNT
        priors.append(counts/counts.sum())
    weights = np.zeros((nodes,nodes))
    for left in range(nodes):
        for right in range(left+1,nodes):
            counts,number = joint_counts(data[:,left],data[:,right],widths[left],widths[right])
            if number < MIN_JOINT:
                continue
            table = counts+PSEUDOCOUNT
            table /= table.sum()
            score = float(np.sum(table*(np.log(table)-np.log(table.sum(axis=1))[:,None]-np.log(table.sum(axis=0))[None,:])))
            if not np.isfinite(score):
                raise FloatingPointError("nonfinite mutual information")
            weights[left,right] = weights[right,left] = max(0,score)
    parents = np.full(nodes,-1,dtype=np.int64)
    keys = np.full(nodes,-np.inf);keys[0] = 0
    visited = np.zeros(nodes,dtype=bool)
    order = []
    for _ in range(nodes):
        node = int(np.argmax(np.where(visited,-np.inf,keys)))
        if not np.isfinite(keys[node]):
            raise ValueError("disconnected candidate graph")
        visited[node] = True;order.append(node)
        better = (~visited) & (weights[node] > keys)
        parents[better] = node;keys[better] = weights[node,better]
    conditionals = [None]*nodes
    children = [[] for _ in range(nodes)]
    for node in order[1:]:
        parent = int(parents[node]);children[parent].append(node)
        counts,number = joint_counts(data[:,parent],data[:,node],widths[parent],widths[node])
        if number >= MIN_JOINT:
            table = counts+PSEUDOCOUNT
            table /= table.sum(axis=1,keepdims=True)
        else:
            table = np.tile(priors[node],(widths[parent],1))
        conditionals[node] = np.log(table)
    return {"widths":list(widths),"parents":parents,"order":order,"children":children,
            "root_log_prior":np.log(priors[0]),"log_conditionals":conditionals,
            "mutual_information":weights,"training_rows":len(data)}


def infer_tree(tree, evidence):
    """All rows and nodes, <=128-row batches; strictly positive missing nodes."""
    evidence = np.asarray(evidence,dtype=np.int64)
    widths,order,children = tree["widths"],tree["order"],tree["children"]
    if evidence.ndim != 2 or evidence.shape[1] != len(widths):
        raise ValueError("evidence shape mismatch")
    for node,width in enumerate(widths):
        if np.any((evidence[:,node] < -1) | (evidence[:,node] >= width)):
            raise ValueError("evidence outside support")
    output = [np.empty((len(evidence),width),dtype=np.float64) for width in widths]
    for start in range(0,len(evidence),BATCH_ROWS):
        stop = min(len(evidence),start+BATCH_ROWS)
        block = evidence[start:stop];size = len(block)
        unary = []
        for node,width in enumerate(widths):
            phi = np.zeros((size,width))
            observed = np.flatnonzero(block[:,node]>=0)
            phi[observed] = -np.inf
            phi[observed,block[observed,node]] = 0
            unary.append(phi)
        total = [None]*len(widths);up = [None]*len(widths)
        for node in reversed(order):
            value = unary[node].copy()
            for child in children[node]:
                value += up[child]
            total[node] = value
            if node:
                message = logsumexp(value[:,None,:]+tree["log_conditionals"][node][None,:,:],axis=2)
                up[node] = message-logsumexp(message,axis=1,keepdims=True)
        down = [None]*len(widths)
        down[0] = np.broadcast_to(tree["root_log_prior"],(size,widths[0]))
        for node in order:
            belief = total[node]+down[node]
            normal = logsumexp(belief,axis=1,keepdims=True)
            if not np.isfinite(normal).all():
                raise FloatingPointError("impossible/nonfinite evidence")
            probability = np.exp(belief-normal)
            # Only unobserved outputs need positive support. Exact evidence
            # retains zero mass on its contradicted known states.
            missing = block[:,node] < 0
            probability[missing] = np.maximum(probability[missing],UNDERFLOW_FLOOR)
            probability /= probability.sum(axis=1,keepdims=True)
            output[node][start:stop] = probability
            for child in children[node]:
                cavity = total[node]+down[node]-up[child]
                message = logsumexp(cavity[:,:,None]+tree["log_conditionals"][child][None,:,:],axis=1)
                down[child] = message-logsumexp(message,axis=1,keepdims=True)
    return output
