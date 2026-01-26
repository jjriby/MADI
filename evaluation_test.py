import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot as dot
import pyagrum as gum
import pyagrum.lib.image as gumimage

from FCI import fci
from PAG import PAG
from utils import draw_pag, pag_edges_as_strings, make_bnlearner_ci

# -------------------------------------------------
# 1) Inspect random BN
# -------------------------------------------------
# That is, we want to look at:
# - number of edges
# - degree histogram
# - variable distributions

print("----------\nTEST 1")
ratios = [i/10 for i in range(10, 21)]
for ratio in ratios:
    print(f"Ratio: {ratio}")
    bn = gum.randomBN(n=10, ratio_arc=ratio, domain_size=2)
    print(f"Connected: {len(bn.connectedComponents()) == 1}")

    print(f"Number of edges: {bn.sizeArcs()}")

    matrix = bn.adjacencyMatrix()
    degrees = {}
    for node in bn.nodes():
        degree = len(np.where(matrix[node])[0])
        if not degree in degrees:
            degrees[degree] = 0
        degrees[degree] += 1

    y = []
    for i in range(max(degrees.keys())):
        if i in degrees:
            y.append(degrees[i])
        else:
            y.append(0)
    print(f"Degree histogram: {y}")

    print(f"Node: {0}; CPT: {bn.cpt(0)}\n")

# -------------------------------------------------
# 2) Inspect the CI algorithms from pyAgrum
# -------------------------------------------------
print("\n----------\nTEST 2")
bn = gum.randomBN(n=10, ratio_arc=1.5, domain_size=2)
dbgen = gum.BNDatabaseGenerator(bn)
dbgen.drawSamples(5_000)
df = dbgen.to_pandas()

# run FCI
alpha = 0.05
test = "chi2"
ci_test = make_bnlearner_ci(df, alpha=alpha, test=test)

X = list(df.columns)
C, sepset = fci(X, ci_test, alpha=alpha)

# run MIIC
learner_miic = gum.BNLearner(df)
learner_miic.useMIIC()
miic_bn = learner_miic.learnBN()

# run GHC
learner_ghc = gum.BNLearner(df)
learner_ghc.useGreedyHillClimbing()
ghc_bn = learner_ghc.learnBN()

print("Drawing in: FCI_test, MIIC_test and GHC_test")
dot = draw_pag(C, filename="FCI_test")
dot.render("FCI_test", format="pdf", cleanup=True)

gumimage.export(bn, "BN_test.pdf")
gumimage.export(miic_bn, "MIIC_test.pdf")
gumimage.export(ghc_bn, "GHC_test.pdf")

# -------------------------------------------------
# 3) Test equivalence class method
# -------------------------------------------------
print("\n----------\nTEST 3")
nodes = ["X0", "X1"]
test_pag = PAG(nodes)
test_pag.add_edge("X0", "X1", "o", "o")

eq = test_pag.eq_class()
print(f"Number of elements in the equivalence class: {len(eq)}")
print("Drawing in test_*")
for i in range(len(eq)):
    dot = draw_pag(eq[i], filename=f"test_{i}")
    dot.render(f"test_{i}", format="pdf", cleanup=True)

# -------------------------------------------------
# 4) Run evaluation on above
# -------------------------------------------------
print("\n----------\nTEST 4")
fci_eq = C.eq_class(add=True)
print("FCI equivalence")
for i in range(min(len(fci_eq), 5)):
    g = fci_eq[i]
    g_bn = g.to_bn(bn)
    if bn.size() == g_bn.size():
        print(f"\t{i}: {gum.ExactBNdistance(bn, g_bn).compute()}")
    else:
        print(f"\t{i}: Not the same size! {bn.size()} and {g_bn.size()}")
    dot = draw_pag(g, filename=f"test_{i}")
    dot.render(f"test_{i}", format="pdf", cleanup=True)

print(f"MIIC equivalence: {gum.ExactBNdistance(bn, miic_bn).compute()}")
print(f"GHC equivalence: {gum.ExactBNdistance(bn, ghc_bn).compute()}")

# -------------------------------------------------
# 5) Look at domain of r.v. in a BN
# -------------------------------------------------
print("\n----------\nTEST 5")
for n in range(10, 15):
    bn = gum.randomBN(n=n, domain_size=5)

    print(f"Size: {n}")
    for node in bn.nodes():
        print(f"\t{node}: {bn.variable(node).domain()}")
    print()