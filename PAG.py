# pag.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """Représentation lisible d'une arête (utile pour debug/visualisation)."""
    u: str
    v: str
    end_u: str
    end_v: str


class PAG:
    """
    PAG = Partial Ancestral Graph (structure de données)
    - Graphe sur des nœuds observés
    - Chaque arête (u,v) porte 2 marques d'extrémité : côté u et côté v
      marque ∈ {"o", "arrow", "tail"}

    Stockage interne:
      _marks[frozenset({u,v})] = {u: mark_u, v: mark_v}
    """

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self._marks = {}  # key=frozenset({u,v}) -> {u: mark_u, v: mark_v}

    # -----------------
    # Arêtes / voisins
    # -----------------
    def has_edge(self, u, v):
        return u != v and frozenset((u, v)) in self._marks

    def add_edge(self, u, v, mark_u="o", mark_v="o"):
        if u == v:
            return
        self._marks[frozenset((u, v))] = {u: mark_u, v: mark_v}

    def remove_edge(self, u, v):
        self._marks.pop(frozenset((u, v)), None)

    def neighbors(self, u):
        nbrs = set()
        for key in self._marks:
            if u in key:
                a, b = tuple(key)
                nbrs.add(a if b == u else b)
        return nbrs

    def adj(self, u):
        return self.neighbors(u)

    # -----------------
    # Marques endpoints
    # -----------------
    def get_end(self, u, v):
        """Retourne la marque au niveau de u sur l'arête (u,v)."""
        return self._marks[frozenset((u, v))][u]

    def set_end(self, u, v, endpoint):
        """Modifie la marque au niveau de u sur l'arête (u,v)."""
        if not self.has_edge(u, v):
            return
        self._marks[frozenset((u, v))][u] = endpoint

    def set_marks(self, u, v, mark_u, mark_v):
        """Fixe les 2 extrémités de l'arête (u,v) en une fois."""
        if not self.has_edge(u, v):
            return
        key = frozenset((u, v))
        self._marks[key][u] = mark_u
        self._marks[key][v] = mark_v

    # -----------------
    # Primitives endpoints (local)
    # -----------------
    def is_o(self, u, v):
        return self.has_edge(u, v) and self.get_end(u, v) == "o"

    def is_arrow(self, u, v):
        return self.has_edge(u, v) and self.get_end(u, v) == "arrow"

    def is_tail(self, u, v):
        return self.has_edge(u, v) and self.get_end(u, v) == "tail"

    def is_star(self, u, v):
        # "*" = pas un cercle
        return self.has_edge(u, v) and self.get_end(u, v) != "o"

    def not_adjacent(self, u, v):
        return not self.has_edge(u, v)

    # -----------------
    # Helpers d'orientation (local)
    # -----------------
    def orient_u_to_v(self, u, v):
        """Force u -> v (tail au niveau de u, arrow au niveau de v)."""
        if not self.has_edge(u, v):
            return False
        before_u = self.get_end(u, v)
        before_v = self.get_end(v, u)
        if before_u == "tail" and before_v == "arrow":
            return False
        self.set_end(u, v, "tail")
        self.set_end(v, u, "arrow")
        return True

    def orient_bidirected(self, u, v):
        """Force u <-> v (arrow aux deux extrémités)."""
        if not self.has_edge(u, v):
            return False
        bu = self.get_end(u, v)
        bv = self.get_end(v, u)
        if bu == "arrow" and bv == "arrow":
            return False
        self.set_end(u, v, "arrow")
        self.set_end(v, u, "arrow")
        return True

    def orient_undirected(self, u, v):
        """Force u — v (tail aux deux extrémités)."""
        if not self.has_edge(u, v):
            return False
        bu = self.get_end(u, v)
        bv = self.get_end(v, u)
        if bu == "tail" and bv == "tail":
            return False
        self.set_end(u, v, "tail")
        self.set_end(v, u, "tail")
        return True

    def set_tail_at(self, u, v):
        """Force juste l’extrémité côté u à tail (u -* v)."""
        if not self.has_edge(u, v):
            return False
        if self.get_end(u, v) == "tail":
            return False
        self.set_end(u, v, "tail")
        return True

    def set_arrow_at(self, u, v):
        """Force juste l’extrémité côté u à arrow (u <-* v)."""
        if not self.has_edge(u, v):
            return False
        if self.get_end(u, v) == "arrow":
            return False
        self.set_end(u, v, "arrow")
        return True

    def is_circle_edge(self, u, v):
        """True ssi l'arête (u,v) est o-o."""
        return self.is_o(u, v) and self.is_o(v, u)

    def edge_allows_forward(self, u, v):
        """
        Condition standard pour "possibly directed" de u vers v :
        il ne doit pas y avoir de tête de flèche au niveau de u sur (u,v).
        """
        return self.has_edge(u, v) and self.get_end(u, v) != "arrow"

    # -----------------
    # Helpers déjà présents
    # -----------------
    def orient_arrow_at(self, u, v):
        """Met une tête de flèche du côté v sur (u,v) => u *-> v (ne modifie pas côté u)."""
        if self.has_edge(u, v):
            self.set_end(v, u, "arrow")

    def orient_tail_at(self, u, v):
        """Met une barre du côté v sur (u,v) => u *- v (ne modifie pas côté u)."""
        if self.has_edge(u, v):
            self.set_end(v, u, "tail")

    def make_undirected_o_o(self, u, v):
        """Force (u,v) en o-o."""
        if self.has_edge(u, v):
            self.set_marks(u, v, "o", "o")

    def reset_all_to_circles(self):
        """Met toutes les arêtes en o-o (Alg 4.3 ligne 17)."""
        for d in self._marks.values():
            for n in list(d.keys()):
                d[n] = "o"

    # -----------------
    # Tests structurels
    # -----------------
    def is_adjacent(self, u, v):
        return self.has_edge(u, v)

    def is_triangle(self, a, b, c):
        """a, b, c forment un triangle si toutes les arêtes existent."""
        return self.has_edge(a, b) and self.has_edge(b, c) and self.has_edge(a, c)

    def is_collider(self, a, b, c):
        """Teste a *-> b <-* c : deux arrowheads pointent vers b."""
        if not (self.has_edge(a, b) and self.has_edge(b, c)):
            return False
        return (self.get_end(b, a) == "arrow") and (self.get_end(b, c) == "arrow")

    # -----------------
    # Itération / debug
    # -----------------
    def edges(self):
        """Liste d'arêtes (utile pour affichage/exports)."""
        out = []
        for key, d in self._marks.items():
            u, v = tuple(key)
            out.append(Edge(u, v, d[u], d[v]))
        return out

    def __str__(self):
        parts = []
        for e in self.edges():
            parts.append(f"{e.u}({e.end_u}) -- {e.v}({e.end_v})")
        return "PAG[" + ", ".join(parts) + "]"