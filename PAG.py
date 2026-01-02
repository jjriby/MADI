from __future__ import annotations
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import pyagrum as gum

Mark = str  # 'o' (circle), '-' (tail), '>' (arrowhead)

class OrderedPAG:
    r"""
    PAG stocké sous forme ordonnée :
      mark(a,b) = marque au bout 'a' sur l'arête a-b.

    Exemples:
      a o-o b : mark(a,b)='o', mark(b,a)='o'
      a --> b : mark(a,b)='-', mark(b,a)='>'
      a <-> b : mark(a,b)='>', mark(b,a)='>'
      a --- b : mark(a,b)='-', mark(b,a)='-'
    """

    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self._m: Dict[Tuple[str, str], Mark] = {}

    def adjacent(self, a: str, b: str) -> bool:
        return (a, b) in self._m

    def add_edge(self, a: str, b: str, ma: Mark = "o", mb: Mark = "o") -> None:
        if self.adjacent(a, b):
            return
        self._m[(a, b)] = ma
        self._m[(b, a)] = mb

    def remove_edge(self, a: str, b: str) -> None:
        self._m.pop((a, b), None)
        self._m.pop((b, a), None)

    def neighbors(self, a: str) -> Set[str]:
        return {b for (x, b) in self._m.keys() if x == a}

    def mark(self, a: str, b: str) -> Mark:
        return self._m[(a, b)]

    def set_mark(self, a: str, b: str, m: Mark) -> bool:
        if not self.adjacent(a, b):
            return False
        old = self._m[(a, b)]
        if old == m:
            return False
        self._m[(a, b)] = m
        return True

    def set_edge(self, a: str, b: str, ma: Mark, mb: Mark) -> bool:
        if not self.adjacent(a, b):
            self.add_edge(a, b, ma, mb)
            return True
        changed = self.set_mark(a, b, ma)
        changed |= self.set_mark(b, a, mb)
        return changed

    # ---- helpers ----
    def is_circle(self, a: str, b: str) -> bool:
        return self.adjacent(a, b) and self.mark(a, b) == "o"

    def is_tail(self, a: str, b: str) -> bool:
        return self.adjacent(a, b) and self.mark(a, b) == "-"

    def is_arrow(self, a: str, b: str) -> bool:
        """a *-> b  <=> mark(b,a) == '>'"""
        return self.adjacent(a, b) and self.mark(b, a) == ">"

    def orient_a_to_b(self, a: str, b: str) -> bool:
        """a --> b"""
        return self.set_edge(a, b, "-", ">")

    def add_arrowhead_at_b(self, a: str, b: str) -> bool:
        """a *-> b : impose '>' au bout b"""
        if not self.adjacent(a, b):
            return False
        return self.set_mark(b, a, ">")

    def add_tail_at_a(self, a: str, b: str) -> bool:
        """a --* b : impose '-' au bout a"""
        if not self.adjacent(a, b):
            return False
        return self.set_mark(a, b, "-")

    def make_undirected(self, a: str, b: str) -> bool:
        """a --- b"""
        return self.set_edge(a, b, "-", "-")

    def make_bidirected(self, a: str, b: str) -> bool:
        """a <-> b"""
        return self.set_edge(a, b, ">", ">")

    def edges_as_strings(self) -> List[str]:
        seen = set()
        out = []
        for a in self.nodes:
            for b in self.neighbors(a):
                if (b, a) in seen:
                    continue
                seen.add((a, b))
                out.append(f"{a} {self.mark(a,b)}-{self.mark(b,a)} {b}")
        return out

