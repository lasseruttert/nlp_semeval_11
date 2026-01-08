import re
import itertools

class SyllogismLogic:
    """
    Fully formal evaluator for canonical categorical syllogisms.
    Exhaustively enumerates all finite models over U={1,2,3}.
    """

    # ---------- PARSER ----------
    @staticmethod
    def parse(sentence: str):
        s = sentence.strip().rstrip(".")
        if s.startswith("All "):
            a, b = re.match(r"All (\w+) are (\w+)", s).groups()
            return ("A", a, b)
        if s.startswith("No "):
            a, b = re.match(r"No (\w+) are (\w+)", s).groups()
            return ("E", a, b)
        if s.startswith("Some "):
            if " are not " in s:
                a, b = re.match(r"Some (\w+) are not (\w+)", s).groups()
                return ("O", a, b)
            a, b = re.match(r"Some (\w+) are (\w+)", s).groups()
            return ("I", a, b)
        raise ValueError(f"Unrecognized form: {sentence}")

    # ---------- SEMANTICS ----------
    @staticmethod
    def holds(form: str, X: set, Y: set) -> bool:
        if form == "A":  # All X are Y
            return X <= Y
        if form == "E":  # No X are Y
            return len(X & Y) == 0
        if form == "I":  # Some X are Y
            return len(X & Y) > 0
        if form == "O":  # Some X are not Y
            return len(X - Y) > 0
        return False

    # ---------- VALIDITY CHECK ----------
    @staticmethod
    def is_valid(major: str, minor: str, conclusion: str) -> bool:
        """
        Checks formal validity by enumerating all possible assignments
        of sets to {A,B,C} over universe {0,1,2}.
        """
        p1 = SyllogismLogic.parse(major)
        p2 = SyllogismLogic.parse(minor)
        c  = SyllogismLogic.parse(conclusion)

        # Determine relevant term letters (up to A,B,C)
        terms = list(set([p1[1], p1[2], p2[1], p2[2], c[1], c[2]]))
        U = {0, 1, 2}
        subsets = [set(x) for i in range(len(U)+1) for x in itertools.combinations(U, i)]

        # Enumerate all combinations of A,B,C subsets
        for As in subsets:
            for Bs in subsets:
                for Cs in subsets:
                    env = {"A": As, "B": Bs, "C": Cs}

                    # If premises hold in this model...
                    if SyllogismLogic.holds(p1[0], env[p1[1]], env[p1[2]]) and \
                       SyllogismLogic.holds(p2[0], env[p2[1]], env[p2[2]]):
                        # ...but conclusion fails → invalid
                        if not SyllogismLogic.holds(c[0], env[c[1]], env[c[2]]):
                            return False
        # No countermodel → valid
        return True
