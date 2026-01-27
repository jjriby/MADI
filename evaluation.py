import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time

import pydot as dot
import pyagrum as gum
import pyagrum.lib.image as gumimage
from pyagrum.lib.bn_vs_bn import GraphicalBNComparator

from func_timeout import func_timeout, FunctionTimedOut

from FCI import fci
from utils import draw_pag, make_bnlearner_ci

# -------------------------------------------------
# Set evaluation parameters
# -------------------------------------------------
num_nodes = [5, 10, 15, 20, 25]
ratio_fraction = [i/10 for i in range(11, 21, 2)]
domain_size = [2, 3, 4]
sample_size = [100, 250, 500, 1_000]

time_limit = 60
alpha = 0.05
test = "chi2"

# -------------------------------------------------
# Run evaluation with full observations
# -------------------------------------------------
def full():
    # define a DataFrame to hold the results
    n_runs = 3 * len(num_nodes) * len(ratio_fraction) * len(domain_size) * len(sample_size) * 5
    results = pd.DataFrame({
        "algorithm": [""] * n_runs,
        "num_nodes": [0] * n_runs,
        "ratio_fraction": [0.0] * n_runs,
        "domain_size": [0] * n_runs,
        "sample_size": [0] * n_runs,
        "trial": [0] * n_runs,
        "time": [0.0] * n_runs,
        "equiv": [False] * n_runs,
        "precision": [0.0] * n_runs,
        "recall": [0.0] * n_runs,
        "f1": [0.0] * n_runs,
        "pure": [0] * n_runs,
        "structural": [0] * n_runs
    })

    # run evaluation
    idx = 0
    for n, r, d, s, trial in itertools.product(num_nodes, ratio_fraction, domain_size, sample_size, range(5)):
        print(f"Running {(n, r, d, s, trial)}")
        # 1. Create random BN
        bn = gum.randomBN(n=n, ratio_arc=r, domain_size=d)
        dbgen = gum.BNDatabaseGenerator(bn)
        dbgen.drawSamples(s)
        df = dbgen.to_pandas()

        ci_test = make_bnlearner_ci(df, alpha=alpha, test=test)

        # 2. Run FCI, MIIC and GHC
        X = list(df.columns)
        start_fci = time.time()
        try:
            C, sepset = func_timeout(time_limit, fci, args=(X, ci_test, alpha))
            fci_time = time.time() - start_fci
        except FunctionTimedOut:
            C = None
            fci_time = time.time() - start_fci
            print(f"FCI timed out on: {(n, r, d, s)})")

        learner_miic = gum.BNLearner(df)
        learner_miic.useMIIC()
        start_miic = time.time()
        try:
            miic_bn = func_timeout(time_limit, learner_miic.learnBN)
            miic_time = time.time() - start_miic
        except FunctionTimedOut:
            miic_bn = None
            miic_time = time.time() - start_miic
            print(f"MIIC timed out on: {(n, r, d, s)})")

        learner_ghc = gum.BNLearner(df)
        learner_ghc.useGreedyHillClimbing()
        start_ghc = time.time()
        try:
            ghc_bn = func_timeout(time_limit, learner_ghc.learnBN)
            ghc_time = time.time() - start_ghc
        except FunctionTimedOut:
            ghc_bn = None
            ghc_time = time.time() - start_ghc
            print(f"GHC timed out on: {(n, r, d, s)})")

        # 3. Compare to true BN
        if miic_bn:
            metrics_miic = compute(bn, miic_bn)
            metrics = list(metrics_miic.values())
            results.loc[3*idx] = ["MIIC", n, r, d, s, trial, miic_time, *metrics]
        if ghc_bn:
            metrics_ghc = compute(bn, ghc_bn)
            metrics = list(metrics_ghc.values())
            results.loc[3*idx+1] = ["GHC", n, r, d, s, trial, ghc_time, *metrics]
        if C:
            fci_eq = C.eq_class(add=True)
            not_same_size = 0
            metrics_fci = {"equiv": False, "precision": -1, "recall": -1, "f1": -1, "pure": -1, "structural": -1}

            for i in range(len(fci_eq)):
                g = fci_eq[i]
                g_bn = g.to_bn(bn)
                
                # we report a single results
                # either an equivalent graph if it exists
                # otherwise, the one with the highest F1
                if bn.size() == g_bn.size():
                    metrics_temp = compute(bn, g_bn)
                    if metrics_temp["equiv"]:
                        metrics_fci = metrics_temp
                    elif metrics_temp["f1"] > metrics_fci["f1"]:
                        metrics_fci = metrics_temp
                else:
                    not_same_size += 1

            metrics = list(metrics_fci.values())
            results.loc[3*idx+2] = ["FCI", n, r, d, s, trial, fci_time, *metrics]
        idx += 1
    results.to_csv("results_full.csv")

def compute(true, pred):
    # 1. Compare to true BN using GraphicalBNComparator
    comparator = GraphicalBNComparator(true, pred)
    results = {}

    results["equiv"] = comparator.equivalentBNs() == "OK"

    scores = comparator.skeletonScores()
    results["precision"] = scores["precision"]
    results["recall"] = scores["recall"]
    results["f1"] = scores["fscore"]

    hamming_scores = comparator.hamming()
    results["pure"] = hamming_scores["hamming"]
    results["structural"] = hamming_scores["structural hamming"]

    return results

full()

# -------------------------------------------------
# Run evaluation with partial observations
# -------------------------------------------------
def partial():
    pass