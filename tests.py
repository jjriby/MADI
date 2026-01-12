import pyagrum as gum
from FCI import fci
from utils import draw_pag, pag_edges_as_strings, make_bnlearner_ci


# -------------------------------------------------
# 1) Génération BN 
# -------------------------------------------------
bn = gum.randomBN(n=10, ratio_arc=1.2, domain_size=2)
dbgen = gum.BNDatabaseGenerator(bn)
dbgen.setRandomVarOrder()
dbgen.drawSamples(5000)
df = dbgen.to_pandas()

# -------------------------------------------------
# 2) CI test via BNLearner (chi2 ou g2)
# -------------------------------------------------
alpha = 0.05
test = "chi2"  # ou "g2"
ci_test = make_bnlearner_ci(df, alpha=alpha, test=test)

# -------------------------------------------------
# 3) Run FCI 
# -------------------------------------------------
X = list(df.columns)
C, sepset = fci(X, ci_test, alpha=alpha)

# -------------------------------------------------
# 4) Visualisation
# -------------------------------------------------
dot = draw_pag(C, filename="pag")
dot.render("pag", format="pdf", cleanup=True)
print("\nPAG saved to pag.pdf")

# -------------------------------------------------
# 5) Affichage 
# -------------------------------------------------
print("\nPAG edges (mark-mark):")
for s in pag_edges_as_strings(C):
    print(s)