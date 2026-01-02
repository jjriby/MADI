from __future__ import annotations
from graphviz import Digraph
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import pyagrum as gum
from PAG import *


def draw_pag(pag, filename="pag"):
    """
    Visualisation d'un PAG (FCI).
    Marques:
      o-o  : cercle-cercle
      -->  : flèche
      <->  : bidirectionnelle
      ---  : non orientée
    """
    dot = Digraph(comment="Partial Ancestral Graph")

    # ajouter noeuds
    for n in pag.nodes:
        dot.node(str(n), str(n))

    seen = set()
    for a in pag.nodes:
        for b in pag.neighbors(a):
            if (b, a) in seen:
                continue
            seen.add((a, b))

            ma = pag.mark(a, b)
            mb = pag.mark(b, a)

            # déterminer le type d'arête
            if ma == "-" and mb == ">":
                dot.edge(str(a), str(b), arrowhead="normal")
            elif ma == ">" and mb == "-":
                dot.edge(str(b), str(a), arrowhead="normal")
            elif ma == ">" and mb == ">":
                dot.edge(str(a), str(b), arrowhead="normal", dir="both")
            elif ma == "-" and mb == "-":
                dot.edge(str(a), str(b), arrowhead="none")
            else:
                # cas avec cercles
                dot.edge(
                    str(a),
                    str(b),
                    arrowhead="odot" if mb == "o" else "normal",
                    arrowtail="odot" if ma == "o" else "none",
                    dir="both",
                )

    return dot

# Path utilities (needed for R4..R10 and PDS)

def is_uncovered_path(pag: OrderedPAG, path: List[str]) -> bool:
    # No chord between non-consecutive nodes
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if j == i + 1:
                continue
            if pag.adjacent(path[i], path[j]):
                return False
    return True


def all_simple_paths(pag: OrderedPAG, start: str, goal: str, max_len: int) -> List[List[str]]:
    paths = []
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) > max_len + 1:
            continue
        if node == goal:
            paths.append(path)
            continue
        for nb in pag.neighbors(node):
            if nb in path:
                continue
            stack.append((nb, path + [nb]))
    return paths


def is_circle_path(pag: OrderedPAG, path: List[str]) -> bool:
    for u, v in zip(path, path[1:]):
        if not (pag.is_circle(u, v) and pag.is_circle(v, u)):
            return False
    return True


def is_possibly_directed_edge(pag: OrderedPAG, u: str, v: str) -> bool:
    # "Possibly directed u -> v" means: no arrowhead into u on that edge
    return pag.adjacent(u, v) and pag.mark(u, v) != ">"


def is_pd_path(pag: OrderedPAG, path: List[str]) -> bool:
    for u, v in zip(path, path[1:]):
        if not is_possibly_directed_edge(pag, u, v):
            return False
    return True


def uncovered_pd_paths(pag: OrderedPAG, a: str, c: str, max_len: int) -> List[List[str]]:
    out = []
    for p in all_simple_paths(pag, a, c, max_len):
        if len(p) >= 2 and is_uncovered_path(pag, p) and is_pd_path(pag, p):
            out.append(p)
    return out


def uncovered_circle_paths(pag: OrderedPAG, a: str, b: str, max_len: int) -> List[List[str]]:
    out = []
    for p in all_simple_paths(pag, a, b, max_len):
        if len(p) >= 3 and is_uncovered_path(pag, p) and is_circle_path(pag, p):
            out.append(p)
    return out


def discriminating_paths_for_B(pag: OrderedPAG, D: str, C: str, B: str, max_len: int) -> List[List[str]]:
    r"""
    Chemins <D,...,A,B,C> discriminants (implémentation pratique).
    Conditions usuelles:
      - D non adjacent C
      - chemin non couvert
      - pour tout V entre D et B : V collider sur le chemin et V *-> C
    """
    if pag.adjacent(D, C):
        return []

    res = []
    for p in all_simple_paths(pag, D, C, max_len):
        if len(p) < 4:
            continue
        if p[-1] != C or p[-2] != B:
            continue
        A = p[-3]
        if not (pag.adjacent(A, B) and pag.adjacent(B, C)):
            continue

        ok = True
        for i in range(1, len(p) - 2):
            V = p[i]
            prev = p[i - 1]
            nxt = p[i + 1]
            # collider prev *-> V <-* nxt
            if not (pag.is_arrow(prev, V) and pag.is_arrow(nxt, V)):
                ok = False
                break
            # and V *-> C
            if not pag.is_arrow(V, C):
                ok = False
                break

        if ok and is_uncovered_path(pag, p):
            res.append(p)

    return res


# Possible-D-SEP (Step B)

def is_collider_on_path(pag: OrderedPAG, a: str, b: str, c: str) -> bool:
    """b is collider on a-b-c iff a *-> b <-* c"""
    return pag.is_arrow(a, b) and pag.is_arrow(c, b)


def forms_triangle(pag: OrderedPAG, a: str, b: str, c: str) -> bool:
    """triangle condition: a adjacent c"""
    return pag.adjacent(a, c)


def possible_d_sep(pag: OrderedPAG, x: str, y: str, max_len: int) -> Set[str]:
    r"""
    Possible-D-SEP(x,y):
    v ∈ PDS(x,y) s'il existe un chemin <x=V0, V1, ..., Vk=v> tel que
    pour chaque triple consécutif <Vi-1, Vi, Vi+1>, Vi est collider sur le chemin
    OU (Vi-1, Vi, Vi+1) forme un triangle.

      - On exclut explicitement x et y de PDS.
    """
    pds: Set[str] = set()

    frontier = [(None, x, 0)]
    visited = set([(None, x)])

    while frontier:
        prev, curr, dist = frontier.pop(0)
        if dist >= max_len:
            continue

        for nb in pag.neighbors(curr):
            if nb == prev:
                continue

            # validate local condition for triple prev-curr-nb
            if prev is not None:
                if not (is_collider_on_path(pag, prev, curr, nb) or forms_triangle(pag, prev, curr, nb)):
                    continue

            state = (curr, nb)
            if state in visited:
                continue
            visited.add(state)

            # Exclude x and y from PDS
            if nb != x and nb != y:
                pds.add(nb)

            frontier.append((curr, nb, dist + 1))

    return pds
