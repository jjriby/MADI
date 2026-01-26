ALGOS:
- FCI (our implementation)
- MIIC (from pyAgrum)
- GHC (from pyAgrum)

REPRESENTATIONS:
- pyAgrum Bayesian Network ```gum.BayesNet()```
- pyAgrum Partial DAG ```gum.PDAG```
- our own Partial Ancestry Graph class ```PAG.PAG```

DATA:
- synthetic data, generated using ```gum.generateBN()```

=> use the randomly generated probability distributions

INPUT PARAMETERS:
- number of variables + number of values
- number of edges + degree histogram, path lengths
- number of dataset points
- probability distributions

ANALYSE RESULTS:
- pyAgrum distance functions
- edges: un-oriented and oriented
- v-structures
- conditional dependances
- computational complexity / execution time
- robustness to number of samples

+ SEE HOW ALGOS HANDLE LATENT VARIABLES

TO DO:
- set random seed in pyAgrum