from PAG import PAG
from utils import k_subsets, unshielded_triples, possible_d_sep
from rules import apply_R1_to_R10


# ------------------------------------------------------------
#   Algorithm 4.1
# ------------------------------------------------------------
def initial_skeleton(X, test, alpha=0.05, S=None, max_l=None, callback=None):
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

    l = -1

    while True:
        l += 1
        if callback is not None:
            callback(C, f"initial_skeleton_l{l}")
        if max_l is not None and l > max_l:
            break

        ordered_pairs = []
        for Xi in C.nodes:
            for Xj in C.adj(Xi):
                if len(C.adj(Xi) - {Xj}) >= l:
                    ordered_pairs.append((Xi, Xj))

        for (Xi, Xj) in ordered_pairs:
            if not C.has_edge(Xi, Xj):
                continue

            candidates = list(C.adj(Xi) - {Xj})
            if len(candidates) < l:
                continue

            for Y in k_subsets(candidates, l):
                if not C.has_edge(Xi, Xj):
                    break

                Y = set(Y)
                cond = set(Y) | S

                if test(Xi, Xj, cond, alpha):
                    C.remove_edge(Xi, Xj)
                    if callback is not None:
                        callback(C, f"initial_skeleton_remove_{Xi}_{Xj}_l{l}")
                    sepset[(Xi, Xj)] = set(Y)
                    sepset[(Xj, Xi)] = set(Y)
                    break

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

    M = unshielded_triples(C)

    return C, sepset, M


# ------------------------------------------------------------
#   Algorithm 4.2
# ------------------------------------------------------------
def orient_v_structures(C, sepset, M, callback=None):
    """
    Met à jour C en orientant les colliders selon Alg 4.2, puis return (C,sepset).
    """
    for (Xi, Xj, Xk) in M:
        S_ik = sepset.get((Xi, Xk), set())
        if Xj not in S_ik:
            if C.has_edge(Xi, Xj):
                C.set_arrow_at(Xj, Xi)
            if C.has_edge(Xk, Xj):
                C.set_arrow_at(Xj, Xk)

            if callback is not None:
                callback(C, f"v_structure_{Xi}_{Xj}_{Xk}")

    return C, sepset


# ------------------------------------------------------------
#   Algorithm 4.3 
# ------------------------------------------------------------
def final_skeleton(C, sepset, test, alpha=0.05, S=None, max_l=None, callback=None):
    """
    Met à jour C et sepset selon Alg 4.3, et renvoie (C, sepset, M).
    """
    if S is None:
        S = set()
    else:
        S = set(S)
        
    for Xi in list(C.nodes):
        pds_Xi = set(possible_d_sep(C, Xi))
        for Xj in list(C.adj(Xi)):
            if not C.has_edge(Xi, Xj):
                continue

            l = -1
            while True:
                l += 1
                if callback is not None:
                    callback(C, f"final_skeleton_{Xi}_{Xj}_l{l}")
                if max_l is not None and l > max_l:
                    break

                candidates = list(pds_Xi - {Xj})
                if len(candidates) < l:
                    break

                removed = False
                for Y in k_subsets(candidates, l):
                    if not C.has_edge(Xi, Xj):
                        removed = True
                        break

                    Y = set(Y)
                    cond = set(Y) | S
                    if test(Xi, Xj, cond, alpha):
                        C.remove_edge(Xi, Xj)
                        if callback is not None:
                            callback(C, f"final_skeleton_remove_{Xi}_{Xj}_l{l}")
                    if callback is not None:
                        callback(C, f"initial_skeleton_remove_{Xi}_{Xj}_l{l}")
                        sepset[(Xi, Xj)] = set(Y)
                        sepset[(Xj, Xi)] = set(Y)
                        removed = True
                        break

                if removed:
                    break

    C.reset_all_to_circles()
    if callback is not None:
        callback(C, "after_reset_all_to_circles")
    M = unshielded_triples(C)
    return C, sepset, M


# ------------------------------------------------------------
#   The FCI algorithm
# ------------------------------------------------------------
def fci(X, test, alpha=0.05, S=None, max_l_skel=None, max_l_pds=None, callback=None):

    C, sepset, M = initial_skeleton(X, test, alpha=alpha, S=S, max_l=max_l_skel, callback=callback)
    if callback is not None:
        callback(C, "after_initial_skeleton")
    C, sepset = orient_v_structures(C, sepset, M, callback=callback)
    if callback is not None:
        callback(C, "after_v_structures_1")
    C, sepset, M = final_skeleton(C, sepset, test, alpha=alpha, S=S, max_l=max_l_pds, callback=callback)
    if callback is not None:
        callback(C, "after_final_skeleton")
    C, sepset = orient_v_structures(C, sepset, M, callback=callback)
    if callback is not None:
        callback(C, "after_v_structures_2")
    C, _ = apply_R1_to_R10(C, sepset, callback=callback)
    if callback is not None:
        callback(C, "after_rules")

    return C, sepset
