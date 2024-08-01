# Simple Reuse of https://github.com/vas-group-imperial/venus/blob/main/src/formula.py


from typing import List


class Formula:
    def __init__(self):
        pass

    def to_nnf(self):
        return self


class TrivialTrueFormula(Formula):
    pass


class TrivialFalseFormula(Formula):
    pass


class BinaryFormula(Formula):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right


class ConjFormula(BinaryFormula):
    def __init__(self, left, right):
        super(ConjFormula, self).__init__(left, right)

    def __str__(self):
        return "AND(" + self.left.__str__() + ", " + self.right.__str__() + ")"

    def to_nnf(self):
        return ConjFormula(self.left.to_nnf(), self.right.to_nnf())


class DisjFormula(BinaryFormula):
    def __init__(self, left, right):
        super(DisjFormula, self).__init__(left, right)

    def __str__(self):
        return "OR(" + self.left.__str__() + ", " + self.right.__str__() + ")"

    def to_nnf(self):
        return DisjFormula(self.left.to_nnf(), self.right.to_nnf())


class NAryFormula(Formula):
    def __init__(self, clauses: List[Formula]):
        super().__init__()
        self.clauses = clauses


class NAryDisjFormula(NAryFormula):
    def __init__(self, clauses):
        super(NAryDisjFormula, self).__init__(clauses)

    def __str__(self):
        return "OR(" + ",".join([clause.__str__() for clause in self.clauses]) + ")"

    def to_nnf(self):
        return NAryDisjFormula([clause.to_nnf() for clause in self.clauses])


class NAryConjFormula(NAryFormula):
    def __init__(self, clauses):
        super(NAryConjFormula, self).__init__(clauses)

    def __str__(self):
        return "AND(" + ",".join([clause.__str__() for clause in self.clauses]) + ")"

    def to_nnf(self):
        return NAryConjFormula([clause.to_nnf() for clause in self.clauses])


class EUntilFormula(BinaryFormula):

    """
    Represents formulas of the form  E phi U^k psi
    """

    def __init__(self, k, left, right):
        super(EUntilFormula, self).__init__(left, right)
        self.k = k

    def __str__(self):
        return (
            "E U{}(".format(self.k)
            + self.left.__str__()
            + ", "
            + self.right.__str__()
            + ")"
        )

    def to_nnf(self):
        left_nnf = self.left.to_nnf()
        right_nnf = self.right.to_nnf()
        subformula = right_nnf
        for i in range(self.k):
            subformula = DisjFormula(
                right_nnf, ConjFormula(left_nnf, ENextFormula(1, subformula))
            )

        return subformula


class AUntilFormula(BinaryFormula):

    """
    Represents formulas of the form  A phi U^k psi
    """

    def __init__(self, k, left, right):
        super(AUntilFormula, self).__init__(left, right)
        self.k = k

    def __str__(self):
        return (
            "A U{}(".format(self.k)
            + self.left.__str__()
            + ", "
            + self.right.__str__()
            + ")"
        )

    def to_nnf(self):
        left_nnf = self.left.to_nnf()
        right_nnf = self.right.to_nnf()

        subformula = right_nnf
        for i in range(self.k):
            subformula = DisjFormula(
                right_nnf, ConjFormula(left_nnf, ANextFormula(1, subformula))
            )

        return subformula


class UnaryFormula(Formula):
    def __init__(self, left):
        super(UnaryFormula, self).__init__()
        self.left = left


class NegationFormula(UnaryFormula):
    def __init__(self, left):
        super(NegationFormula, self).__init__(left)

    def __str__(self):
        return "NOT(" + self.left.__str__() + ")"

    def to_nnf(self):
        subformula = self.left
        if isinstance(subformula, TrivialFalseFormula):
            return TrivialTrueFormula

        if isinstance(subformula, TrivialTrueFormula):
            return TrivialFalseFormula

        if isinstance(subformula, NegationFormula):
            return subformula.left.to_nnf()

        if isinstance(subformula, VarVarConstraint):
            return VarVarConstraint(
                subformula.op1,
                inverted_sense[subformula.sense],
                subformula.op2,
                subformula.offset,
            )

        if isinstance(subformula, VarConstConstraint):
            return VarConstConstraint(
                subformula.op1, inverted_sense[subformula.sense], subformula.op2
            )

        if isinstance(subformula, LinExprConstraint):
            return LinExprConstraint(
                subformula.op1, inverted_sense[subformula.sense], subformula.op2
            )

        if isinstance(subformula, ConjFormula):
            return DisjFormula(
                NegationFormula(subformula.left).to_nnf(),
                NegationFormula(subformula.right).to_nnf(),
            )

        if isinstance(subformula, DisjFormula):
            return ConjFormula(
                NegationFormula(subformula.left).to_nnf(),
                NegationFormula(subformula.right).to_nnf(),
            )

        if isinstance(subformula, NAryDisjFormula):
            return NAryConjFormula(
                [NegationFormula(clause).to_nnf() for clause in subformula.clauses]
            )

        if isinstance(subformula, NAryConjFormula):
            return NAryDisjFormula(
                [NegationFormula(clause).to_nnf() for clause in subformula.clauses]
            )

        if isinstance(subformula, ENextFormula):
            return ANextFormula(subformula.k, NegationFormula(subformula.left).to_nnf())

        if isinstance(subformula, ANextFormula):
            return ENextFormula(subformula.k, NegationFormula(subformula.left).to_nnf())

        if isinstance(subformula, EUntilFormula):
            """
            NOT(E phi1 U^k phi2) =>
                    AG^k NOT(phi2) OR A NOT(phi2) U^k (NOT(phi2) AND NOT(phi1))
            """
            left_nnf = NegationFormula(subformula.left).to_nnf()
            right_nnf = NegationFormula(subformula.right).to_nnf()

            # left
            left_subformula = right_nnf
            for i in range(subformula.k):
                left_subformula = ConjFormula(
                    right_nnf, ANextFormula(1, left_subformula)
                )

            right_subformula = AUntilFormula(
                subformula.k, right_nnf, ConjFormula(right_nnf, left_nnf)
            ).to_nnf()

            return DisjFormula(left_subformula, right_subformula)

        if isinstance(subformula, AUntilFormula):
            """
            NOT(A phi1 U^k phi2) =>
                    AG^k NOT(phi2) OR E NOT(phi2) U^k (NOT(phi2) AND NOT(phi1))
            """
            left_nnf = NegationFormula(subformula.left).to_nnf()
            right_nnf = NegationFormula(subformula.right).to_nnf()

            left_subformula = right_nnf
            for i in range(subformula.k):
                left_subformula = ConjFormula(
                    right_nnf, ENextFormula(1, left_subformula)
                )

            right_subformula = EUntilFormula(
                subformula.k, right_nnf, ConjFormula(right_nnf, left_nnf)
            ).to_nnf()

            return DisjFormula(left_subformula, right_subformula)

        return NegationFormula(subformula.to_nnf())


class ENextFormula(UnaryFormula):

    """
    Represents formulas of the form  E X^k phi
    """

    def __init__(self, k, left):
        super(ENextFormula, self).__init__(left)
        self.k = k

    def __str__(self):
        return "E X{}(".format(self.k) + self.left.__str__() + ")"

    def to_nnf(self):
        return ENextFormula(self.k, self.left.to_nnf())


class ANextFormula(UnaryFormula):
    """
    Represents formulas of the form  A X^k phi
    """

    def __init__(self, k, left):
        super(ANextFormula, self).__init__(left)
        self.k = k

    def __str__(self):
        return "A X{}(".format(self.k) + self.left.__str__() + ")"

    def to_nnf(self):
        return ANextFormula(self.k, self.left.to_nnf())


(LT, GT, NE) = ("<", ">", "!=")
(LE, GE, EQ) = ("<=", ">=", "==")
inverted_sense = {LE: GT, GT: LE, GE: LT, LT: GE, EQ: NE, NE: EQ}


class StateCoordinate:
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return "({})".format(self.i)


class LinearExpression:
    def __init__(self, coord_coeff_map):
        self.coord_coeff_map = coord_coeff_map

    def __str__(self):
        return " + ".join(
            ["{}*({})".format(self.coord_coeff_map[i], i) for i in self.coord_coeff_map]
        )


class Constraint(Formula):
    def __init__(self, op1, sense, op2):
        super(Constraint, self).__init__()
        self.op1 = op1
        self.op2 = op2
        self.sense = sense


class VarVarConstraint(Constraint):
    def __init__(self, op1, sense, op2, offset=0):
        super(VarVarConstraint, self).__init__(op1, sense, op2)
        self.offset = offset

    def __str__(self):
        return (
            self.op1.__str__() + self.sense + self.op2.__str__() + self.offset.__str__()
        )


class VarConstConstraint(Constraint):
    def __init__(self, op1, sense, op2):
        super(VarConstConstraint, self).__init__(op1, sense, op2)

    def __str__(self):
        return self.op1.__str__() + self.sense + "{}".format(self.op2)


class LinExprConstraint(Constraint):
    def __init__(self, op1, sense, op2):
        super(LinExprConstraint, self).__init__(op1, sense, op2)

    def __str__(self):
        return self.op1.__str__() + self.sense + self.op2.__str__()


class EmptyFormula:
    def __init__(self):
        pass


def transform_to_leq_constraint(constraint_formula: Constraint):
    if constraint_formula.sense in {GT, GE}:
        return constraint_formula.__class__(
            op1=constraint_formula.op2,
            sense=inverted_sense[constraint_formula.sense],
            op2=constraint_formula.op1,
        )
    else:
        return constraint_formula


def get_eg_formula(n, phi, psi):
    formula = psi
    for i in range(n):
        formula = ConjFormula(phi, ENextFormula(1, formula))
    return formula


def get_ag_formula(n, phi, psi):
    formula = psi
    for i in range(n):
        formula = ConjFormula(phi, ANextFormula(1, formula))
    return formula


def get_ef_formula(n, phi, psi):
    formula = psi
    for i in range(n):
        formula = DisjFormula(phi, ENextFormula(1, formula))
    return formula


# obsolete with the new NAry Conj and Disj classes
def get_conjunction(atoms: List[Formula]):
    formula = atoms[0]
    for i in range(1, len(atoms)):
        formula = ConjFormula(atoms[i], formula)
    return formula


def get_disjunction(atoms: List[Formula]):
    formula = atoms[0]
    for i in range(1, len(atoms)):
        formula = DisjFormula(atoms[i], formula)
    return formula
