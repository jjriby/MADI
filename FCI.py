from PAG import PAG
from utils import k_subsets, unshielded_triples, possible_d_sep
from rules import apply_R1_to_R10


# ------------------------------------------------------------
#   Algorithm 4.1
# ------------------------------------------------------------
def initial_skeleton(X, test, alpha=0.05, S=None, max_l=None):
    """
    Retourne (C, sepset, M) comme l'algo 4.1.
    """
    if S is None:
        S = set()
    else:
        S = set(S)

    # 1: Form the complete graph C on X with edges o-o
    C = PAG(X)
    nodes = list(C.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            C.add_edge(nodes[i], nodes[j], "o", "o")

    # sepset est un dict : sepset[(Xi,Xj)] = Y (set)
    sepset = {}

    # 2: Let ℓ = -1
    l = -1

    # 3: repeat
    while True:
        # 4: Let ℓ = ℓ + 1
        l += 1
        if max_l is not None and l > max_l:
            break

        # 5: repeat  (boucle sur les paires ordonnées)
        # until all ordered pairs ... with |adj(C,Xi)\{Xj}| >= ℓ have been considered
        # On l'implémente en parcourant toutes les paires ordonnées valides dans l'état courant.
        ordered_pairs = []
        for Xi in C.nodes:
            for Xj in C.adj(Xi):
                if len(C.adj(Xi) - {Xj}) >= l:
                    ordered_pairs.append((Xi, Xj))

        for (Xi, Xj) in ordered_pairs:
            # l'arête a pu être supprimée par une itération précédente
            if not C.has_edge(Xi, Xj):
                continue

            # 11: Choose a (new) set Y ⊆ adj(C,Xi)\{Xj} with |Y| = ℓ
            candidates = list(C.adj(Xi) - {Xj})
            if len(candidates) < l:
                continue

            # 10: repeat ... until Xi and Xj no longer adjacent or all Y considered
            for Y in k_subsets(candidates, l):
                if not C.has_edge(Xi, Xj):
                    break

                Y = set(Y)
                cond = set(Y) | S

                # 12: if Xi and Xj conditionally independent given Y ∪ S then
                if test(Xi, Xj, cond, alpha):
                    # 13: Delete edge Xi o-o Xj
                    C.remove_edge(Xi, Xj)
                    # 14: sepset(Xi,Xj)=sepset(Xj,Xi)=Y
                    sepset[(Xi, Xj)] = set(Y)
                    sepset[(Xj, Xi)] = set(Y)
                    break

        # 18: until all pairs adjacent satisfy |adj(C,Xi)\{Xj}| <= ℓ
        done = True
        for Xi in C.nodes:
            for Xj in C.adj(Xi):
                if len(C.adj(Xi) - {Xj}) > l:
                    done = False
                    break
            if not done:
                break

        if done:
            break

    # 19: Form list M of all unshielded triples ... (middle unspecified in paper)
    # On renvoie des triples (Xi, Xj, Xk) avec Xj au milieu (utile pour Alg 4.2).
    M = unshielded_triples(C)

    # 20: return C, sepset, M
    return C, sepset, M


# ------------------------------------------------------------
#   Algorithm 4.2
# ------------------------------------------------------------
def orient_v_structures(C, sepset, M):
    """
    Met à jour C en orientant les colliders selon Alg 4.2, puis return (C,sepset).
    """
    # 1: for all elements <Xi,Xj,Xk> of M do
    for (Xi, Xj, Xk) in M:
        # 2: if Xj not in sepset(Xi,Xk) then
        S_ik = sepset.get((Xi, Xk), set())
        if Xj not in S_ik:
            # 3: Orient Xi *-o Xj o-* Xk as Xi *-> Xj <-* Xk
            # => mettre arrow au niveau de Xj sur (Xi,Xj) et (Xk,Xj)
            if C.has_edge(Xi, Xj):
                C.set_arrow_at(Xj, Xi)
            if C.has_edge(Xk, Xj):
                C.set_arrow_at(Xj, Xk)

    # 6: return C, sepset
    return C, sepset


# ------------------------------------------------------------
#   Algorithm 4.3 
# ------------------------------------------------------------
def final_skeleton(C, sepset, test, alpha=0.05, S=None, max_l=None):
    """
    Met à jour C et sepset selon Alg 4.3, et renvoie (C, sepset, M).
    """
    if S is None:
        S = set()
    else:
        S = set(S)

    # 1: for all vertices Xi in C do
    for Xi in list(C.nodes):
        # 2: Compute pds(C, Xi, .)
        pds_Xi = set(possible_d_sep(C, Xi))

        # 3: for all vertices Xj ∈ adj(C, Xi) do
        for Xj in list(C.adj(Xi)):
            # si l'arête a déjà sauté
            if not C.has_edge(Xi, Xj):
                continue

            # 4: Let ℓ = -1
            l = -1

            # 5: repeat
            while True:
                # 6: Let ℓ = ℓ + 1
                l += 1
                if max_l is not None and l > max_l:
                    break

                # 8: Choose a (new) set Y ⊆ pds(C,Xi,.)\{Xj} with |Y|=ℓ
                candidates = list(pds_Xi - {Xj})
                if len(candidates) < l:
                    break

                removed = False

                # 7: repeat ... until adjacency gone or all Y considered
                for Y in k_subsets(candidates, l):
                    if not C.has_edge(Xi, Xj):
                        removed = True
                        break

                    Y = set(Y)
                    cond = set(Y) | S

                    # 9: if Xi and Xj conditionally independent given Y ∪ S then
                    if test(Xi, Xj, cond, alpha):
                        # 10: Delete edge Xi ** Xj from C
                        C.remove_edge(Xi, Xj)
                        # 11: sepset(Xi,Xj)=sepset(Xj,Xi)=Y
                        sepset[(Xi, Xj)] = set(Y)
                        sepset[(Xj, Xi)] = set(Y)
                        removed = True
                        break

                # 13/14: until Xi and Xj no longer adjacent OR all Y considered / |pds\{Xj}| < ℓ
                if removed:
                    break

            # fin boucle l
        # fin boucle Xj
    # fin boucle Xi

    # 17: Reorient all edges in C as o-o
    C.reset_all_to_circles()

    # 18: Form a list M of all unshielded triples ...
    M = unshielded_triples(C)

    # 19: return C, sepset, M
    return C, sepset, M


# ------------------------------------------------------------
#   The FCI algorithm
# ------------------------------------------------------------
def fci(X, test, alpha=0.05, S=None, max_l_skel=None, max_l_pds=None):

    C, sepset, M = initial_skeleton(X, test, alpha=alpha, S=S, max_l=max_l_skel)
    C, sepset = orient_v_structures(C, sepset, M)
    C, sepset, M = final_skeleton(C, sepset, test, alpha=alpha, S=S, max_l=max_l_pds)
    C, sepset = orient_v_structures(C, sepset, M)
    C, _ = apply_R1_to_R10(C, sepset)

    return C, sepset