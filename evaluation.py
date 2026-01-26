import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot as dot
import pyagrum as gum
import pyagrum.lib.image as gumimage

from func_timeout import func_timeout, FunctionTimedOut

from FCI import fci
from utils import draw_pag, make_bnlearner_ci

# -------------------------------------------------
# Set evaluation parameters
# -------------------------------------------------
num_nodes = [5, 10, 15, 20, 25]
ratio_fractions = [i/10 for i in range(11, 21)]
domain_size = [2, 3, 4, 5]
sample_size = [100, 250, 500, 1_000, 2_500, 5_000]

time_limit = 60
alpha = 0.05
test = "chi2"

# -------------------------------------------------
# Run evaluation
# -------------------------------------------------
for n, r, d, s in zip(num_nodes, ratio_fractions, domain_size, sample_size):
    # 1. Create 5 random BNs
    for trial in range(5):
        bn = gum.randomBN(n=n, ratio_arc=r, domain_size=d)
        dbgen = gum.BNDatabaseGenerator(bn)
        dbgen.drawSamples(s)
        df = dbgen.to_pandas()

        ci_test = make_bnlearner_ci(df, alpha=alpha, test=test)

        # 2. Run FCI, MIIC and GHC
        X = list(df.columns)
        try:
            C, sepset = func_timeout(time_limit, fci, args=(X, ci_test, alpha))
        except FunctionTimedOut:
            print(f"FCI timed out on: {(n, r, d, s)})")

        learner_miic = gum.BNLearner(df)
        learner_miic.useMIIC()
        try:
            miic_bn = func_timeout(time_limit, learner_miic.learnBN)
        except FunctionTimedOut:
            print(f"MIIC timed out on: {(n, r, d, s)})")

        learner_ghc = gum.BNLearner(df)
        learner_ghc.useGreedyHillClimbing()
        try:
            ghc_bn = func_timeout(time_limit, learner_ghc.learnBN)
        except FunctionTimedOut:
            print(f"GHC timed out on: {(n, r, d, s)})")
        