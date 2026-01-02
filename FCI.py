from __future__ import annotations
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from PAG import *
from utils import *
import pyagrum as gum


# FCI: Step A + Step B + Orientation (R0..R10)
class FCI:
    def __init__(
        self,
        data,
        alpha: float = 0.05,
        test: str = "chi2",
        max_cond_set_A: Optional[int] = None,  # Step A
        max_cond_set_B: int = 3,               # Step B
        pds_max_path_len: int = 6,
        rule_max_path_len: int = 6,
        max_pds_candidates: int = 30,
    ):
        
        self.data = data
        self.alpha = alpha
        self.test = test.lower()

        self.max_cond_set_A = max_cond_set_A
        self.max_cond_set_B = max_cond_set_B
        self.pds_max_path_len = pds_max_path_len
        self.rule_max_path_len = rule_max_path_len
        self.max_pds_candidates = max_pds_candidates

        self.learner = gum.BNLearner(data)

        self.vars = list(self.data.columns)
        self.adj: Dict[str, Set[str]] = {}
        self.sepset: Dict[Tuple[str, str], FrozenSet[str]] = {}
        self.pag = OrderedPAG(self.vars)

    # --- Xi test ---
    def xi_indep(self, x: str, y: str, cond: List[str]) -> bool:
        # Safety: remove x,y and duplicates
        cond = [v for v in cond if v != x and v != y]
        # dedupe while preserving order
        cond = list(dict.fromkeys(cond))

        if self.test == "g2":
            _, pval = self.learner.G2(x, y, cond)
        else:
            _, pval = self.learner.chi2(x, y, cond)
        return pval >= self.alpha

    # Step A: skeleton PC-like
    def skeleton(self) -> None:
        V = self.vars
        self.adj = {v: set(V) - {v} for v in V}

        if self.max_cond_set_A is None:
            self.max_cond_set_A = max(0, len(V) - 2)

        for l in range(0, self.max_cond_set_A + 1):
            edges = [(x, y) for x in V for y in self.adj[x] if x < y]
            changed = False

            for x, y in edges:
                cand = list(self.adj[x] - {y})
                if len(cand) < l:
                    continue

                for S in combinations(cand, l):
                    if self.xi_indep(x, y, list(S)):
                        self.adj[x].discard(y)
                        self.adj[y].discard(x)
                        s = frozenset(S)
                        self.sepset[(x, y)] = s
                        self.sepset[(y, x)] = s
                        changed = True
                        break

            if not changed:
                break

        # init PAG with o-o edges for current skeleton
        self.pag = OrderedPAG(self.vars)
        for x in V:
            for y in self.adj[x]:
                if x < y:
                    self.pag.add_edge(x, y, "o", "o")


    # Step B: Possible-D-SEP retests
    def possible_dsep_prune(self) -> None:
        V = self.vars
        edges = [(x, y) for x in V for y in self.adj[x] if x < y]

        for x, y in edges:
            if y not in self.adj[x]:
                continue

            pds = list(possible_d_sep(self.pag, x, y, self.pds_max_path_len))

            pds = [v for v in pds if v != x and v != y]
            pds = list(dict.fromkeys(pds))

            if len(pds) > self.max_pds_candidates:
                pds = pds[: self.max_pds_candidates]

            removed = False
            for l in range(0, min(self.max_cond_set_B, len(pds)) + 1):
                for S in combinations(pds, l):
                    cond = list(S)
                    if self.xi_indep(x, y, cond):
                        self.adj[x].discard(y)
                        self.adj[y].discard(x)
                        self.pag.remove_edge(x, y)

                        s = frozenset(cond)
                        self.sepset[(x, y)] = s
                        self.sepset[(y, x)] = s

                        removed = True
                        break
                if removed:
                    break


    # Orientation: Rules R0 to R10
    def rule_R0_colliders(self) -> bool:
        changed = False
        for y in self.vars:
            neigh = list(self.pag.neighbors(y))
            for x, z in combinations(neigh, 2):
                if self.pag.adjacent(x, z):
                    continue
                sep = self.sepset.get((x, z), frozenset())
                if y not in sep:
                    changed |= self.pag.add_arrowhead_at_b(x, y)
                    changed |= self.pag.add_arrowhead_at_b(z, y)
        return changed

    def rule_R1(self) -> bool:
        changed = False
        for B in self.vars:
            for A in self.pag.neighbors(B):
                if not self.pag.is_arrow(A, B):
                    continue
                for C in self.pag.neighbors(B):
                    if C == A:
                        continue
                    if self.pag.adjacent(A, C):
                        continue
                    if not self.pag.is_circle(B, C):
                        continue
                    changed |= self.pag.orient_a_to_b(B, C)
        return changed

    def rule_R2(self) -> bool:
        changed = False
        for A in self.vars:
            for C in self.pag.neighbors(A):
                if not self.pag.is_circle(A, C):
                    continue
                for B in self.pag.neighbors(A):
                    if B == C or not self.pag.adjacent(B, C):
                        continue
                    cond1 = (self.pag.is_tail(A, B) and self.pag.is_arrow(A, B) and self.pag.is_arrow(B, C))
                    cond2 = (self.pag.is_arrow(A, B) and self.pag.is_tail(B, C) and self.pag.is_arrow(B, C))
                    if cond1 or cond2:
                        changed |= self.pag.add_arrowhead_at_b(A, C)
        return changed

    def rule_R3(self) -> bool:
        changed = False
        V = self.vars
        for B in V:
            neigh = list(self.pag.neighbors(B))
            for A, C in combinations(neigh, 2):
                if not (self.pag.is_arrow(A, B) and self.pag.is_arrow(C, B)):
                    continue
                if self.pag.adjacent(A, C):
                    continue
                for D in V:
                    if D in (A, B, C):
                        continue
                    if not (self.pag.adjacent(A, D) and self.pag.adjacent(D, C) and self.pag.adjacent(D, B)):
                        continue
                    if not (self.pag.is_circle(A, D) and self.pag.is_circle(C, D) and self.pag.is_circle(D, B)):
                        continue
                    changed |= self.pag.orient_a_to_b(D, B)
        return changed

    def rule_R4(self) -> bool:
        changed = False
        for D in self.vars:
            for C in self.vars:
                if D == C or self.pag.adjacent(D, C):
                    continue
                for B in self.pag.neighbors(C):
                    if B == D:
                        continue
                    if not self.pag.is_circle(B, C):
                        continue
                    paths = discriminating_paths_for_B(self.pag, D, C, B, self.rule_max_path_len)
                    for p in paths:
                        A = p[-3]
                        sep = self.sepset.get((D, C), frozenset())
                        if B in sep:
                            changed |= self.pag.orient_a_to_b(B, C)
                        else:
                            changed |= self.pag.make_bidirected(A, B)
                            changed |= self.pag.make_bidirected(B, C)
        return changed

    def rule_R5(self) -> bool:
        changed = False
        for A in self.vars:
            for B in self.pag.neighbors(A):
                if A >= B:
                    continue
                if not (self.pag.is_circle(A, B) and self.pag.is_circle(B, A)):
                    continue
                for u in uncovered_circle_paths(self.pag, A, B, self.rule_max_path_len):
                    if len(u) < 4:
                        continue
                    C = u[1]
                    D = u[-2]
                    if self.pag.adjacent(A, D):
                        continue
                    if self.pag.adjacent(B, C):
                        continue
                    changed |= self.pag.make_undirected(A, B)
                    for x, y in zip(u, u[1:]):
                        changed |= self.pag.make_undirected(x, y)
        return changed

    def rule_R6_R7(self) -> bool:
        changed = False
        for B in self.vars:
            for A in self.pag.neighbors(B):
                for C in self.pag.neighbors(B):
                    if C == A:
                        continue
                    if not self.pag.is_circle(B, C):
                        continue
                    if self.pag.is_tail(A, B) and self.pag.is_tail(B, A):  # A---B
                        changed |= self.pag.add_tail_at_a(B, C)
                    if (self.pag.is_tail(A, B) and self.pag.is_circle(B, A) and (not self.pag.adjacent(A, C))):
                        changed |= self.pag.add_tail_at_a(B, C)
        return changed

    def rule_R8(self) -> bool:
        changed = False
        for A in self.vars:
            for C in self.pag.neighbors(A):
                if not (self.pag.is_circle(A, C) and self.pag.is_arrow(A, C)):  # Ao->C
                    continue
                for B in self.vars:
                    if B in (A, C):
                        continue
                    if not (self.pag.adjacent(A, B) and self.pag.adjacent(B, C)):
                        continue
                    cond1 = (self.pag.is_tail(A, B) and self.pag.is_arrow(A, B) and
                             self.pag.is_tail(B, C) and self.pag.is_arrow(B, C))
                    cond2 = (self.pag.is_tail(A, B) and self.pag.is_circle(B, A) and
                             self.pag.is_tail(B, C) and self.pag.is_arrow(B, C))
                    if cond1 or cond2:
                        changed |= self.pag.orient_a_to_b(A, C)
        return changed

    def rule_R9(self) -> bool:
        changed = False
        for A in self.vars:
            for C in self.pag.neighbors(A):
                if not (self.pag.is_circle(A, C) and self.pag.is_arrow(A, C)):  # Ao->C
                    continue
                for u in uncovered_pd_paths(self.pag, A, C, self.rule_max_path_len):
                    if len(u) < 3:
                        continue
                    B = u[1]
                    if self.pag.adjacent(C, B):
                        continue
                    changed |= self.pag.orient_a_to_b(A, C)
        return changed

    def rule_R10(self) -> bool:
        changed = False
        for A in self.vars:
            for C in self.pag.neighbors(A):
                if not (self.pag.is_circle(A, C) and self.pag.is_arrow(A, C)):  # Ao->C
                    continue
                parents = []
                for X in self.pag.neighbors(C):
                    if X == A:
                        continue
                    if self.pag.is_tail(X, C) and self.pag.is_arrow(X, C):  # X-->C
                        parents.append(X)
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        B, D = parents[i], parents[j]
                        pAB = uncovered_pd_paths(self.pag, A, B, self.rule_max_path_len)
                        pAD = uncovered_pd_paths(self.pag, A, D, self.rule_max_path_len)
                        for u1 in pAB:
                            if len(u1) < 2:
                                continue
                            M = u1[1]
                            for u2 in pAD:
                                if len(u2) < 2:
                                    continue
                                N = u2[1]
                                if M == N:
                                    continue
                                if self.pag.adjacent(M, N):
                                    continue
                                changed |= self.pag.orient_a_to_b(A, C)
        return changed

    def orient_R1_to_R4(self) -> None:
        while True:
            changed = False
            changed |= self.rule_R1()
            changed |= self.rule_R2()
            changed |= self.rule_R3()
            changed |= self.rule_R4()
            if not changed:
                break

    def orient_R1_to_R10(self) -> None:
        while True:
            changed = False
            changed |= self.rule_R1()
            changed |= self.rule_R2()
            changed |= self.rule_R3()
            changed |= self.rule_R4()
            changed |= self.rule_R5()
            changed |= self.rule_R6_R7()
            changed |= self.rule_R8()
            changed |= self.rule_R9()
            changed |= self.rule_R10()
            if not changed:
                break

    # RUN FCI
    def run(self) -> OrderedPAG:
        # Step A
        self.skeleton()

        # Pre-orientation (helps PDS step)
        self.rule_R0_colliders()
        self.orient_R1_to_R4()

        # Step B
        self.possible_dsep_prune()

        # Final orientation
        self.rule_R0_colliders()
        self.orient_R1_to_R10()

        return self.pag


# Exemple d'utilisation
if __name__ == "__main__":

    bn = gum.randomBN(n=10, ratio_arc=1.2, domain_size=2)
    dbgen = gum.BNDatabaseGenerator(bn)
    dbgen.setRandomVarOrder()
    dbgen.drawSamples(5000)
    df = dbgen.to_pandas()

    fci = FCI(
        df,
        alpha=0.05,
        test="chi2",
        max_cond_set_A=3,
        max_cond_set_B=3,
        pds_max_path_len=6,
        rule_max_path_len=6,
        max_pds_candidates=30,
    )
    pag = fci.run()

    dot = draw_pag(pag)
    dot.render("pag_result", view=True)
    print("\nPAG edges (mark-mark):")
    for e in pag.edges_as_strings():
        print(e)
