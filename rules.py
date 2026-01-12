from utils import find_uncovered_circle_path, find_uncovered_pd_path, find_discriminating_path


def rule_R1(C):
    changed = False
    # R1: if α*→β o-*γ and α and γ not adjacent, orient β o-*γ as β→γ
    for beta in C.nodes:
        for alpha in C.adj(beta):
            if not (C.is_arrow(beta, alpha) and C.is_star(alpha, beta)):  # alpha *-> beta
                continue
            for gamma in C.adj(beta):
                if gamma == alpha:
                    continue
                if not (C.is_o(beta, gamma) and C.is_star(gamma, beta)):  # beta o-* gamma
                    continue
                if C.not_adjacent(alpha, gamma):
                    changed |= C.orient_u_to_v(beta, gamma)
    return changed


def rule_R2(C):
    changed = False
    # R2: If (α→β*→γ OR α*→β→γ) and α*-oγ then orient α*-oγ as α*→γ
    for alpha in C.nodes:
        for gamma in C.adj(alpha):
            # α *-o γ : star côté alpha, circle côté gamma
            if not (C.is_star(alpha, gamma) and C.is_o(gamma, alpha)):
                continue

            for beta in C.nodes:
                if beta == alpha or beta == gamma:
                    continue

                # Case1: α→β and β*→γ
                case1 = (
                    C.has_edge(alpha, beta)
                    and C.is_tail(alpha, beta)
                    and C.is_arrow(beta, alpha)  # arrow at beta
                    and C.has_edge(beta, gamma)
                    and C.is_arrow(gamma, beta)  # arrow at gamma => beta *-> gamma
                    and C.is_star(beta, gamma)   # star at beta
                )

                # Case2: α*→β and β→γ
                case2 = (
                    C.has_edge(alpha, beta)
                    and C.is_arrow(beta, alpha)  # alpha *-> beta (arrow at beta)
                    and C.is_star(alpha, beta)
                    and C.has_edge(beta, gamma)
                    and C.is_tail(beta, gamma)
                    and C.is_arrow(gamma, beta)  # beta -> gamma (arrow at gamma)
                )

                if case1 or case2:
                    # α*-oγ devient α*→γ : on met arrow au niveau de gamma (côté gamma)
                    changed |= C.set_arrow_at(gamma, alpha)
    return changed


def rule_R3(C):
    changed = False
    # R3: If α*→β<-*γ, α*-oθ o-*γ, α and γ not adjacent, and θ*-oβ,
    #     then orient θ*-oβ as θ*→β.
    for beta in C.nodes:
        for alpha in C.adj(beta):
            if not (C.is_arrow(beta, alpha) and C.is_star(alpha, beta)):  # alpha *-> beta
                continue

            for gamma in C.adj(beta):
                if gamma == alpha:
                    continue
                if not (C.is_arrow(beta, gamma) and C.is_star(gamma, beta)):  # gamma *-> beta
                    continue
                if not C.not_adjacent(alpha, gamma):
                    continue

                for theta in C.nodes:
                    if theta in (alpha, beta, gamma):
                        continue

                    # alpha *-o theta
                    if not (C.has_edge(alpha, theta) and C.is_star(alpha, theta) and C.is_o(theta, alpha)):
                        continue
                    # theta o-* gamma
                    if not (C.has_edge(theta, gamma) and C.is_o(theta, gamma) and C.is_star(gamma, theta)):
                        continue
                    # theta *-o beta
                    if not (C.has_edge(theta, beta) and C.is_star(theta, beta) and C.is_o(beta, theta)):
                        continue

                    # θ*-oβ -> θ*→β : mettre arrow au niveau de beta
                    changed |= C.set_arrow_at(beta, theta)
    return changed


def rule_R4(C, sepset):
    changed = False
    # R4: If u=<θ,...,α,β,γ> discriminating path between θ and γ for β, and β o-* γ;
    #     then if β ∈ Sepset(θ,γ) orient β o-* γ as β→γ;
    #     else orient (α,β,γ) as α↔β↔γ.
    for beta in C.nodes:
        for gamma in C.adj(beta):
            if not (C.is_o(beta, gamma) and C.is_star(gamma, beta)):  # beta o-* gamma
                continue

            for alpha in C.adj(beta):
                if alpha == gamma:
                    continue

                path = find_discriminating_path(C, beta, gamma, alpha)
                if path is None:
                    continue
                theta = path[0]

                sep = sepset.get((theta, gamma), set())
                if beta in sep:
                    changed |= C.orient_u_to_v(beta, gamma)
                else:
                    changed |= C.orient_bidirected(alpha, beta)
                    changed |= C.orient_bidirected(beta, gamma)
    return changed


def rule_R5(C):
    changed = False
    # R5: For every remaining α o-o β, if there is an uncovered circle path
    #     p=(α,γ,...,θ,β) between α and β s.t. α,θ not adjacent and β,γ not adjacent,
    #     then orient α o-o β and every edge on p as undirected (—).
    for a in C.nodes:
        for b in C.adj(a):
            if a >= b:
                continue
            if not C.is_circle_edge(a, b):
                continue

            p = find_uncovered_circle_path(C, a, b)
            if p is None or len(p) < 4:
                continue

            gamma = p[1]
            theta = p[-2]
            if C.has_edge(a, theta):
                continue
            if C.has_edge(b, gamma):
                continue

            changed |= C.orient_undirected(a, b)
            for i in range(len(p) - 1):
                changed |= C.orient_undirected(p[i], p[i + 1])
    return changed


def rule_R6(C):
    changed = False
    # R6: If α—β o-* γ then orient β o-* γ as β-* γ (mettre tail au niveau de β)
    for beta in C.nodes:
        for alpha in C.adj(beta):
            if not (C.is_tail(alpha, beta) and C.is_tail(beta, alpha)):  # alpha—beta
                continue
            for gamma in C.adj(beta):
                if gamma == alpha:
                    continue
                if C.is_o(beta, gamma) and C.is_star(gamma, beta):  # beta o-* gamma
                    changed |= C.set_tail_at(beta, gamma)
    return changed


def rule_R7(C):
    changed = False
    # R7: If α—oβ o-*γ, and α and γ not adjacent, then orient β o-* γ as β-* γ
    for beta in C.nodes:
        for alpha in C.adj(beta):
            if not (C.is_tail(alpha, beta) and C.is_o(beta, alpha)):  # alpha—o beta
                continue
            for gamma in C.adj(beta):
                if gamma == alpha:
                    continue
                if not (C.is_o(beta, gamma) and C.is_star(gamma, beta)):
                    continue
                if C.not_adjacent(alpha, gamma):
                    changed |= C.set_tail_at(beta, gamma)
    return changed


def rule_R8(C):
    changed = False
    # R8: If (α→β→γ OR α—β→γ) and α o→ γ, orient α o→ γ as α→γ
    for alpha in C.nodes:
        for gamma in C.adj(alpha):
            if not (C.is_o(alpha, gamma) and C.is_arrow(gamma, alpha)):  # alpha o-> gamma
                continue

            for beta in C.nodes:
                if beta in (alpha, gamma):
                    continue

                # beta -> gamma
                cond_bg = (
                    C.has_edge(beta, gamma)
                    and C.is_tail(beta, gamma)
                    and C.is_arrow(gamma, beta)
                )
                if not cond_bg:
                    continue

                cond1 = (
                    C.has_edge(alpha, beta)
                    and C.is_tail(alpha, beta)
                    and C.is_arrow(beta, alpha)  # alpha -> beta
                )
                cond2 = (
                    C.has_edge(alpha, beta)
                    and C.is_tail(alpha, beta)
                    and C.is_tail(beta, alpha)   # alpha — beta
                )

                if cond1 or cond2:
                    changed |= C.set_tail_at(alpha, gamma)  # alpha o-> gamma => alpha -> gamma
    return changed


def rule_R9(C):
    changed = False
    # R9: If α o→ γ and p=(α,β,θ,...,γ) is an uncovered p.d. path from α to γ
    #     such that γ and β are not adjacent, then orient α o→ γ as α → γ.
    for alpha in C.nodes:
        for gamma in C.adj(alpha):
            if not (C.is_o(alpha, gamma) and C.is_arrow(gamma, alpha)):  # alpha o-> gamma
                continue

            p = find_uncovered_pd_path(C, alpha, gamma)
            if p is None or len(p) < 3:
                continue

            beta = p[1]
            if C.not_adjacent(gamma, beta):
                changed |= C.set_tail_at(alpha, gamma)
    return changed


def rule_R10(C):
    changed = False
    # R10: Suppose α o→ γ, β → γ ← θ, p1 uncovered p.d. path from α to β, p2 from α to θ.
    #      Let μ neighbor of α on p1, ω neighbor of α on p2.
    #      If μ and ω distinct, and are not adjacent, then orient α o→ γ as α → γ.
    for alpha in C.nodes:
        for gamma in C.adj(alpha):
            if not (C.is_o(alpha, gamma) and C.is_arrow(gamma, alpha)):  # alpha o-> gamma
                continue

            candidates = []
            for v in C.adj(gamma):
                if v == alpha:
                    continue
                # v -> gamma
                if C.is_tail(v, gamma) and C.is_arrow(gamma, v):
                    candidates.append(v)

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    beta = candidates[i]
                    theta = candidates[j]

                    p1 = find_uncovered_pd_path(C, alpha, beta)
                    p2 = find_uncovered_pd_path(C, alpha, theta)
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
                        continue

                    mu = p1[1]
                    omega = p2[1]
                    if mu == omega:
                        continue

                    if C.not_adjacent(mu, omega):
                        changed |= C.set_tail_at(alpha, gamma)
    return changed


def apply_R1_to_R10(C, sepset):
    """Applique R1..R10 jusqu’à convergence."""
    changed_any = False
    changed = True
    while changed:
        changed = False

        changed |= rule_R1(C)
        changed |= rule_R2(C)
        changed |= rule_R3(C)
        changed |= rule_R4(C, sepset)
        changed |= rule_R5(C)
        changed |= rule_R6(C)
        changed |= rule_R7(C)
        changed |= rule_R8(C)
        changed |= rule_R9(C)
        changed |= rule_R10(C)

        if changed:
            changed_any = True

    return C, changed_any