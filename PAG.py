from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass

import pyagrum as gum

@dataclass(frozen=True)
class Edge:
    """Représentation d'une arête ."""
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
    """

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self._marks = {} 
        self.latent_id = 0

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
    # Primitives endpoints
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
    # Helpers d'orientation
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
    
    # -----------------
    # Equivalence class
    # -----------------
    def has_cycle(self):
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.neighbors(node):
                if self.is_tail(node, neighbor) and self.is_arrow(neighbor, node):
                    if neighbor not in visited:
                        if dfs(neighbor):
                            return True
                        
                    elif neighbor in rec_stack:
                        return True
            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
            
        return False
    
    def is_dag(self):
        for u in self.nodes:
            for v in self.nodes:
                if self.is_o(u, v) or self.is_o(v, u):
                    return False
                if self.is_arrow(u, v) and self.is_arrow(v, u):
                    return False
        return True

    def eq_class(self, add=False):
        """ create a queue of PAGs
        we transform each one step closer to a DAG by:
         - orienting an edge with an o
         - adding a latent variable for bi-directed edges """
        processing = [deepcopy(self)]
        out = []

        while processing:
            curr = processing.pop(0)

            if curr.has_cycle():
                continue
            if curr.is_dag():
                out.append(curr)
                continue
            
            done = False
            for u in curr.nodes:
                if done:
                    break
                for v in curr.nodes:
                    if curr.has_edge(u, v):
                        if curr.is_o(u, v):
                            copy_arrow = deepcopy(curr)
                            copy_arrow.set_end(u, v, "arrow")
                            processing.append(copy_arrow)

                            if not curr.is_tail(v, u):
                                copy_tail = deepcopy(curr)
                                copy_tail.set_end(u, v, "tail")
                                processing.append(copy_tail)
                            done = True
                            break
                        
                        elif curr.is_arrow(u, v) and curr.is_arrow(v, u):
                            if add:
                                latent = f"L{self.latent_id}"
                                self.latent_id += 1
                                copy_latent = deepcopy(curr)
                                copy_latent.nodes.append(latent)

                                copy_latent.remove_edge(u, v)
                                
                                copy_latent.add_edge(latent, u, "tail", "arrow")
                                copy_latent.add_edge(latent, v, "tail", "arrow")
                                processing.append(copy_latent)

                            else:
                                # don't add latent notes to be able to compare the BNs
                                copy_latent = deepcopy(curr)
                                copy_latent.remove_edge(u, v)
                                processing.append(copy_latent)

                            done = True
                            break
        return out
    
    # -----------------
    # Transform PAG to BN
    # -----------------
    def to_bn(self, bn_ref):
        bn = gum.BayesNet()

        # add nodes to BN
        # (use size 2 for variable domains since it doesn't matter for evaluation)
        for node in self.nodes:
            if node in bn_ref.nodes():
                var = bn_ref.variable(node)
                bn.add(gum.RangeVariable(node, node, var.minVal(), var.maxVal()))
            else:
                bn.add(gum.RangeVariable(node, node, 0, 1))

        # add edges to BN
        for u in self.nodes:
            for v in self.nodes:
                if self.has_edge(u, v):
                    if self.is_arrow(u, v):
                        bn.addArc(u, v)

        return bn