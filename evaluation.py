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
# Run evaluation with full observations
# -------------------------------------------------
def full():
    num_nodes = [5, 10, 15, 20, 25]
    ratio_fraction = [i/10 for i in range(11, 21, 2)]
    domain_size = [2, 3, 4]
    sample_size = [100, 250, 500, 1_000]

    time_limit = 60
    alpha = 0.05
    test = "chi2"

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
            metrics_fci = {"equiv": False, "precision": 0.0, "recall": 0.0, "f1": 0.0, "pure": 0, "structural": 0}

            for i in range(len(fci_eq)):
                g = fci_eq[i]
                g_bn = g.to_bn(bn)
                
                # we report a single result
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

def analyse_full():
    results = pd.read_csv("results_full.csv")

    # 1. For each model, when is it able to find the true structure
    true_miic = results.loc[(results["equiv"]) & (results["algorithm"]=="MIIC"),
                            ["num_nodes", "ratio_fraction", "domain_size", "sample_size", "trial"]]
    print(f"MIIC was correct in {len(true_miic)} cases.")

    true_ghc = results.loc[(results["equiv"]) & (results["algorithm"]=="GHC"),
                            ["num_nodes", "ratio_fraction", "domain_size", "sample_size", "trial"]]
    print(f"GHC was correct in {len(true_ghc)} cases.")

    true_fci = results.loc[(results["equiv"]) & (results["algorithm"]=="FCI"),
                            ["num_nodes", "ratio_fraction", "domain_size", "sample_size", "trial"]]
    print(f"FCI was correct in {len(true_fci)} cases.")

    # 2. Make 5 plots (on the same figure, one for each metric) of the three models for:
    # - as num_nodes varies, best other parameters (ratio, domain, sample)
    # - etc. (all other possible combinations)
    metrics = ["precision", "recall", "f1", "pure", "structural"]
    parameters = ["num_nodes", "ratio_fraction", "domain_size", "sample_size"]
    algorithms = ["MIIC", "GHC", "FCI"]
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    fig.suptitle("Algorithm Performance Across Parameters", fontsize=16, y=0.995)
    
    colors = {"MIIC": "blue", "GHC": "green", "FCI": "red"}
    markers = {"MIIC": "o", "GHC": "s", "FCI": "^"}
    
    for row, metric in enumerate(metrics):
        for col, param in enumerate(parameters):
            ax = axes[row, col]
            
            # For each parameter, we want to vary it and keep others at their best values
            # We'll aggregate across trials and optimize over other parameters
            other_params = [p for p in parameters if p != param]
            
            for algo in algorithms:
                algo_data = results[results["algorithm"] == algo].copy()
                
                # Group by the parameter we're varying
                param_values = sorted(algo_data[param].unique())
                mean_values = []
                
                for val in param_values:
                    subset = algo_data[algo_data[param] == val]
                    
                    # For each value of this parameter, find the best combination of other parameters
                    # by taking the mean across trials and finding the ma
                    grouped = subset.groupby(other_params)[metric].mean()
                    
                    # Get all trials for the best parameter combination
                    best_params = grouped.idxmax()
                    if not isinstance(best_params, tuple):
                        best_params = (best_params,)
                    
                    # Filter to get all trials with these best parameters
                    mask = pd.Series(True, index=subset.index)
                    for i, other_param in enumerate(other_params):
                        mask &= (subset[other_param] == best_params[i])
                    
                    best_subset = subset[mask]
                    mean_values.append(best_subset[metric].mean())
                   
                # Plot with error bars
                ax.plot(param_values, mean_values, 
                        label=algo, color=colors[algo], marker=markers[algo],
                        linewidth=2, markersize=6)
            
            # Formatting
            ax.set_xlabel(param.replace("_", " ").title(), fontsize=10)
            if col == 0:
                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            
            # Set appropriate y-limits based on metric
            if metric in ["precision", "recall", "f1"]:
                ax.set_ylim(-0.05, 1.05)
            
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved plot to algorithm_comparison.png")

# -------------------------------------------------
# Run evaluation with partial observations
# -------------------------------------------------
# Test whether FCI is capable of detecting latent variables
# (look at different num_nodes, ratio_fraction and sample_size for domain_size=2)
def partial():
    num_nodes = [5, 10, 15, 20, 25]
    ratio_fraction = [i/10 for i in range(11, 21, 2)]
    domain_size = [2, 3, 4]
    sample_size = [100, 250, 500]

    time_limit = 60
    alpha = 0.05
    test = "chi2"

    n_runs = sum(num_nodes) * len(ratio_fraction) * len(domain_size) * len(sample_size) * 5
    results = pd.DataFrame({
        "num_nodes": [0] * n_runs,
        "node_removed": [None] * n_runs,
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
        "structural": [0] * n_runs,
        "avg_found": [0.0] * n_runs
    })

    # Test masking a single node
    idx = 0
    for n, r, d, s, trial in itertools.product(num_nodes, ratio_fraction, domain_size, sample_size, range(5)):
        print(f"Running {(n, r, d, s, trial)}")
        # 1. Create random BN
        bn = gum.randomBN(n=n, ratio_arc=r, domain_size=d)
        dbgen = gum.BNDatabaseGenerator(bn)
        dbgen.drawSamples(s)
        df_full = dbgen.to_pandas()

        # 2. Create a dataset with a node removed, for each node
        names = list(bn.names())
        for i in range(len(names)):
            name = names[i]
            df = df_full.drop(columns=[name])
            ci_test = make_bnlearner_ci(df, alpha=alpha, test=test)

            X = list(df.columns)
            start_fci = time.time()
            try:
                C, sepset = func_timeout(time_limit, fci, args=(X, ci_test, alpha))
                fci_time = time.time() - start_fci
            except FunctionTimedOut:
                C = None
                fci_time = time.time() - start_fci
                print(f"FCI timed out on: {(n, r, d, s)})")

            # 3. Compare to true bn
            if C:
                fci_eq = C.eq_class(add=True)
                metrics_fci = {"equiv": False, "precision": 0.0, "recall": 0.0, "f1": 0.0, "pure": 0, "structural": 0}

                found = 0
                for j in range(len(fci_eq)):
                    g = fci_eq[j]
                    g_bn = g.to_bn(bn)
                    found = 0
                
                    # we report a single result
                    # either an equivalent graph if it exists
                    # otherwise, the one with the highest F1
                    if bn.size() == g_bn.size():
                        metrics_temp = compute(bn, g_bn)
                        if metrics_temp["equiv"]:
                            metrics_fci = metrics_temp
                        elif metrics_temp["f1"] > metrics_fci["f1"]:
                            metrics_fci = metrics_temp
                    else:
                        found += (g_bn.size() - len(C.nodes))

                if metrics_fci["equiv"]:
                    metrics_fci["avg_found"] = 1
                # might have no DAGs in the equivalence class
                elif len(fci_eq) > 0:
                    metrics_fci["avg_found"] = found / len(fci_eq)
                else:
                    metrics_fci["avg_found"] = 0
            else:
                metrics_fci = {"equiv": None, "precision": None, "recall": None, "f1": None, "pure": None, "structural": None, "avg_found": None}

            metrics = list(metrics_fci.values())
            results.loc[idx+i] = [n, name, r, d, s, trial, fci_time, *metrics]
        idx += len(names)
    results.to_csv("results_partial.csv")

partial()

def analyse_partial():
    pass