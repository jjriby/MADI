# utils.py
from __future__ import annotations
from itertools import combinations
from collections import deque
from graphviz import Digraph
import pyagrum as gum


def k_subsets(lst, k):
    """Itère sur tous les sous-ensembles de taille k de lst."""
    return combinations(lst, k)


def unshielded_triples(C):
    triples = set()
    for Xj in C.nodes:
        nbrs = list(C.adj(Xj))
        for Xi, Xk in combinations(nbrs, 2):
            if not C.has_edge(Xi, Xk):
                triples.add((Xi, Xj, Xk))
    return list(triples)


def possible_d_sep(C, Xi, Xj=None):
    """
    Définition 3.3 :
      Xk ∈ pds(C, Xi, Xj) ssi il existe un chemin π entre Xi et Xk tel que
      pour tout sous-chemin consécutif <Xm, Xl, Xh> de π :
         - Xl est un collider sur ce sous-chemin dans C, OU
         - <Xm, Xl, Xh> forme un triangle dans C.

    Remarque 3.1 :
      Xj ne joue pas de rôle, donc l'argument Xj est ignoré (mais conservé pour l'API).
    """
    seen_states = {(None, Xi)}   # (prev, cur)
    queue = [(None, Xi)]
    reached = {Xi}

    while queue:
        prev, cur = queue.pop(0)

        for nxt in C.adj(cur):
            if nxt == prev:
                continue

            # Condition sur chaque triple consécutif <prev,cur,nxt>
            ok = True
            if prev is not None:
                if not (C.is_collider(prev, cur, nxt) or C.is_triangle(prev, cur, nxt)):
                    ok = False

            if not ok:
                continue

            st = (cur, nxt)
            if st in seen_states:
                continue

            seen_states.add(st)
            reached.add(nxt)
            queue.append((cur, nxt))

    # Dans l'algo 4.3 on enlève ensuite {Xi, Xj} de toute façon.
    reached.discard(Xi)
    return reached

def is_uncovered_path(C, path):
    for i in range(1, len(path) - 1):
        if C.has_edge(path[i - 1], path[i + 1]):
            return False
    return True


def _pd_allows_step(C, vi, vj):
    if not C.has_edge(vi, vj):
        return False
    return (C.get_end(vi, vj) != "arrow") and (C.get_end(vj, vi) != "tail")


def is_potentially_directed_path(C, path):
    for i in range(len(path) - 1):
        if not _pd_allows_step(C, path[i], path[i + 1]):
            return False
    return True


def find_uncovered_pd_path(C, v0, vn, max_len=25):
    """
    Cherche un chemin p=(v0,...,vn) qui est:
      - uncovered
      - potentially directed de v0 vers vn
    """
    q = deque()
    q.append([v0])

    while q:
        path = q.popleft()
        if len(path) > max_len:
            continue

        last = path[-1]

        if last == vn and len(path) >= 2:
            if is_uncovered_path(C, path) and is_potentially_directed_path(C, path):
                return path
            continue

        for nxt in C.adj(last):
            if nxt in path:
                continue

            # Def 10 : compatible avec une p.d. path dans le sens last -> nxt
            if not _pd_allows_step(C, last, nxt):
                continue

            newp = path + [nxt]

            # prune uncovered local : éviter un triple shielded immédiatement
            if len(newp) >= 3 and C.has_edge(newp[-3], newp[-1]):
                continue

            q.append(newp)

    return None


def find_uncovered_circle_path(C, a, b, max_len=25):
    """
    uncovered circle path entre a et b.
    Un "circle path" = toutes les arêtes du chemin sont o-o.
    """
    q = deque()
    q.append([a])

    while q:
        path = q.popleft()
        if len(path) > max_len:
            continue

        last = path[-1]
        if last == b and len(path) >= 2:
            if not is_uncovered_path(C, path):
                continue
            ok = True
            for i in range(len(path) - 1):
                if not (C.is_o(path[i], path[i + 1]) and C.is_o(path[i + 1], path[i])):
                    ok = False
                    break
            if ok:
                return path
            continue

        for nxt in C.adj(last):
            if nxt in path:
                continue

            # circle edge only: o-o
            if not (C.is_o(last, nxt) and C.is_o(nxt, last)):
                continue

            newp = path + [nxt]
            if len(newp) >= 3 and C.has_edge(newp[-3], newp[-1]):
                continue

            q.append(newp)

    return None


def find_discriminating_path(C, beta, gamma, alpha, max_len=25):
    """
    Cherche un chemin p = <X, ..., W, V, Y> discriminating pour V
    """
    if not (C.has_edge(alpha, beta) and C.has_edge(beta, gamma)):
        return None

    q = deque()
    q.append([alpha])

    while q:
        chain = q.popleft()
        if len(chain) > max_len:
            continue

        X = chain[0]
        path = chain + [beta, gamma]

        # (i) au moins 3 arêtes => au moins 4 sommets
        if len(path) >= 4:
            # (iii.a) X non adjacent à Y(=gamma)
            if C.has_edge(X, gamma):
                pass
            else:
                ok = True
                idx_beta = len(path) - 2
                between = path[1:idx_beta]  # inclut alpha

                for i in range(1, idx_beta):
                    Z = path[i]
                    prev = path[i - 1]
                    nxt = path[i + 1]

                    # collider sur le chemin: prev *-> Z <-* nxt (arrowheads vers Z)
                    if not C.is_collider(prev, Z, nxt):
                        ok = False
                        break

                    # parent de gamma: Z -> gamma
                    # dans nos marks: tail au niveau de Z, arrow au niveau de gamma
                    if not (C.has_edge(Z, gamma) and C.is_tail(Z, gamma) and C.is_arrow(gamma, Z)):
                        ok = False
                        break

                if ok:
                    return path

        cur = chain[0]
        next_node = chain[1] if len(chain) >= 2 else beta

        # Pour réduire l'explosion, si cur n'est PAS parent de gamma, inutile d'étendre.
        if not (C.has_edge(cur, gamma) and C.is_tail(cur, gamma) and C.is_arrow(gamma, cur)):
            continue

        for newX in C.adj(cur):
            if newX in chain:
                continue

            if not C.is_collider(newX, cur, next_node):
                continue

            if C.has_edge(newX, gamma):
                continue

            q.append([newX] + chain)

    return None


def draw_pag(C, filename="pag"):
    """
    Visualisation d'un PAG (FCI) avec Graphviz.

    Conventions PAG:
      - o     : cercle
      - arrow : flèche
      - tail  : barre

    Rendu:
      o-o    : cercles
      tail->arrow : flèche
      arrow-arrow : bidirectionnelle
      tail-tail   : non orientée
    """
    dot = Digraph(name=filename, comment="Partial Ancestral Graph")

    # noeuds
    for n in C.nodes:
        dot.node(str(n), str(n))

    seen = set()
    for u in C.nodes:
        for v in C.adj(u):
            if (v, u) in seen:
                continue
            seen.add((u, v))

            mu = C.get_end(u, v)
            mv = C.get_end(v, u)

            # cas simples
            if mu == "tail" and mv == "arrow":
                dot.edge(str(u), str(v), arrowhead="normal")
            elif mu == "arrow" and mv == "tail":
                dot.edge(str(v), str(u), arrowhead="normal")
            elif mu == "arrow" and mv == "arrow":
                dot.edge(str(u), str(v), arrowhead="normal", dir="both")
            elif mu == "tail" and mv == "tail":
                dot.edge(str(u), str(v), arrowhead="none")
            else:
                # cas avec cercles (o)
                dot.edge(
                    str(u),
                    str(v),
                    dir="both",
                    arrowhead="odot" if mv == "o" else "normal",
                    arrowtail="odot" if mu == "o" else "none",
                )

    return dot



def make_bnlearner_ci(data, alpha=0.05, test="chi2"):

    learner = gum.BNLearner(data)
    def ci_test(x, y, cond_set, a=alpha):
        # Safety: remove x,y + dedupe
        cond = list(cond_set) if cond_set else []
        cond = [v for v in cond if v != x and v != y]
        cond = list(dict.fromkeys(cond))

        if test.lower() == "g2":
            _, pval = learner.G2(x, y, cond)
        else:
            _, pval = learner.chi2(x, y, cond)

        return pval >= a

    return ci_test

def pag_edges_as_strings(C):
    """
    Affiche les arêtes sous forme lisible:
    u(end_u) -- v(end_v)
    """
    out = []
    for e in C.edges():
        out.append(f"{e.u}({e.end_u}) -- {e.v}({e.end_v})")
    return out