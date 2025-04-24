from functools import partial
from typing import List

from airobas.blocks_hub.mip_blocks_lib.commons.formula import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    ConjFormula,
    Constraint,
    DisjFormula,
    Formula,
    LinExprConstraint,
    NAryConjFormula,
    NAryDisjFormula,
    NegationFormula,
    StateCoordinate,
    VarConstConstraint,
    VarVarConstraint,
    transform_to_leq_constraint,
)
from airobas.blocks_hub.mip_blocks_lib.commons.neural_network import NeuralNetwork
from gurobipy import GRB, Model, Var, quicksum


def create_formula_for_local_robustness(label: int, output_dim: int, offset=0):
    coordinates = [StateCoordinate(i) for i in range(output_dim)]
    atoms = [
        VarVarConstraint(coordinates[i], LT, coordinates[label], offset) for i in range(output_dim) if i not in [label]
    ]
    return NAryConjFormula(atoms)


def instance_satisfies_property(neural_network: NeuralNetwork, output_formula: Formula):
    output_bounds = neural_network.layers[-1].bounds["out"]
    lower_bounds = output_bounds["l"]
    upper_bounds = output_bounds["u"]
    return check_formula_satisfaction(output_formula, lower_bounds, upper_bounds)


def return_functions_properties(output_formula: Formula):
    def f(x: NeuralNetwork, formula: Formula):
        return check_formula_satisfaction(
            formula=formula,
            lower_bounds=x.layers[-1].bounds["out"]["l"],
            upper_bounds=x.layers[-1].bounds["out"]["u"],
        )

    neg_formula = NegationFormula(output_formula)
    return {
        "sat-for-sure": partial(f, formula=output_formula),
        "unsat-for-sure": partial(f, formula=neg_formula),
    }


def check_formula_satisfaction(formula: Formula, lower_bounds, upper_bounds):
    if isinstance(formula, Constraint):
        sense = formula.sense
        if sense == LT:
            if isinstance(formula, VarVarConstraint):
                return upper_bounds[formula.op1.i] < lower_bounds[formula.op2.i] + formula.offset
            if isinstance(formula, VarConstConstraint):
                return upper_bounds[formula.op1.i] < formula.op2
        elif sense == GT:
            if isinstance(formula, VarVarConstraint):
                return lower_bounds[formula.op1.i] > upper_bounds[formula.op2.i] + formula.offset
            if isinstance(formula, VarConstConstraint):
                return lower_bounds[formula.op1.i] > formula.op2

    if isinstance(formula, ConjFormula):
        return check_formula_satisfaction(formula.left, lower_bounds, upper_bounds) and check_formula_satisfaction(
            formula.right, lower_bounds, upper_bounds
        )

    if isinstance(formula, NAryConjFormula):
        for clause in formula.clauses:
            if not check_formula_satisfaction(clause, lower_bounds, upper_bounds):
                return False
        return True

    if isinstance(formula, DisjFormula):
        return check_formula_satisfaction(formula.left, lower_bounds, upper_bounds) or check_formula_satisfaction(
            formula.right, lower_bounds, upper_bounds
        )

    if isinstance(formula, NAryDisjFormula):
        for clause in formula.clauses:
            if check_formula_satisfaction(clause, lower_bounds, upper_bounds):
                return True
        return False


def get_output_constrs_specific(
    output_formula: Formula,
    model: Model,
    neural_net: NeuralNetwork,
    output_vars: List[Var],
):
    cll = None
    if isinstance(output_formula, VarConstConstraint):
        const = output_formula.op2
        var: StateCoordinate = output_formula.op1
        sense = output_formula.sense
        if sense in {LT, LE}:
            model.setObjective(output_vars[var.i], sense=GRB.MAXIMIZE)
            model.update()

            def callback(gmodel, where):
                if where == GRB.Callback.MIPNODE:
                    status = gmodel.cbGet(GRB.Callback.MIPNODE_STATUS)
                    if status == GRB.OPTIMAL:
                        obj = gmodel.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        if obj <= const:
                            print("Finished")
                            gmodel.terminate()
                if where == GRB.Callback.MIPSOL:
                    obj = gmodel.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > const:
                        print("Finished by finding a counter example.")
                        gmodel.terminate()

            return callback
        if sense in {GT, GE}:
            model.setObjective(output_vars[var.i], sense=GRB.MINIMIZE)
            model.update()

            def callback(gmodel, where):
                if where == GRB.Callback.MIPNODE:
                    status = gmodel.cbGet(GRB.Callback.MIPNODE_STATUS)
                    if status == GRB.OPTIMAL:
                        obj = gmodel.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        if obj >= const:
                            print("Finished, property ok")
                            gmodel.terminate()
                if where == GRB.Callback.MIPSOL:
                    obj = gmodel.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < const:
                        print("Finished by finding a counter example.")

            return callback

    elif isinstance(output_formula, NAryDisjFormula) or isinstance(output_formula, NAryConjFormula):
        deltas = []
        lbs = []
        ubs = []
        for j in range(len(output_formula.clauses)):
            if isinstance(output_formula.clauses[j], VarVarConstraint):
                formula = transform_to_leq_constraint(output_formula.clauses[j])
                lb = (
                    neural_net.layers[-1].bounds["out"]["l"][formula.op1.i]
                    - neural_net.layers[-1].bounds["out"]["u"][formula.op2.i]
                )
                ub = (
                    neural_net.layers[-1].bounds["out"]["u"][formula.op1.i]
                    - neural_net.layers[-1].bounds["out"]["l"][formula.op2.i]
                )
                deltas += [model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"delta_{j}")]
                model.addConstr(deltas[j] == output_vars[formula.op1.i] - output_vars[formula.op2.i])
            if isinstance(output_formula.clauses[j], VarConstConstraint):
                formula = transform_to_leq_constraint(output_formula.clauses[j])
                if isinstance(formula.op1, StateCoordinate):
                    lb = neural_net.layers[-1].bounds["out"]["l"][formula.op1.i] - formula.op2
                    ub = neural_net.layers[-1].bounds["out"]["u"][formula.op1.i] - formula.op2
                    deltas += [model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"delta_{j}")]
                    model.addConstr(deltas[j] == output_vars[formula.op1.i] - formula.op2)
                if isinstance(formula.op2, StateCoordinate):
                    lb = formula.op1 - neural_net.layers[-1].bounds["out"]["u"][formula.op2.i]
                    ub = formula.op1 - neural_net.layers[-1].bounds["out"]["l"][formula.op2.i]
                    deltas += [model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"delta_{j}")]
                    model.addConstr(deltas[j] == formula.op1 - output_vars[formula.op2.i])
            lbs += [lb]
            ubs += [ub]
        if isinstance(output_formula, NAryDisjFormula):
            aggregated_delta = model.addVar(lb=min(lbs), ub=min(ubs), name=f"agg_delta")
            model.addGenConstrMin(aggregated_delta, deltas, name="min_const")
            # we should find a negative aggregated delta !
            if min(lbs) > 0:
                return "counter_example."  # TODO : handle this
            # aggregated delta should remain NEGATIVE (so OBJBound<0)
            model.setObjective(aggregated_delta, sense=GRB.MAXIMIZE)
            model.update()

        if isinstance(output_formula, NAryConjFormula):
            aggregated_delta = model.addVar(lb=max(lbs), ub=max(ubs), name=f"agg_delta")
            model.addGenConstrMax(aggregated_delta, deltas, name="max_const")
            model.setObjective(aggregated_delta, sense=GRB.MAXIMIZE)
            model.update()
            if max(lbs) > 0:
                return "counter_example."  # TODO : handle this
            # aggregated delta should remain NEGATIVE (so OBJBound<0)

        def callback(m, where):
            if where == GRB.Callback.MIPNODE:
                status = m.cbGet(GRB.Callback.MIPNODE_STATUS)
                if status == GRB.OPTIMAL:
                    obj = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
                    if obj < 0:
                        print("satisfied")
                        m.terminate()
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > 0:
                    print("Finished by failure = ")
                    m.terminate()

        return callback
    else:
        get_output_constrs(output_formula, model, output_vars)


def get_callback_specific(output_formula: Formula):
    cll = None
    if isinstance(output_formula, VarConstConstraint):
        const = output_formula.op2
        var: StateCoordinate = output_formula.op1
        sense = output_formula.sense
        if sense in {LT, LE}:

            def callback(gmodel, where):
                if where == GRB.Callback.MIPNODE:
                    status = gmodel.cbGet(GRB.Callback.MIPNODE_STATUS)
                    if status == GRB.OPTIMAL:
                        obj = gmodel.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        if obj <= const:
                            print("Finished")
                            gmodel.terminate()
                if where == GRB.Callback.MIPSOL:
                    obj = gmodel.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > const:
                        print("Finished by finding a counter example.")
                        gmodel.terminate()

            return callback
        if sense in {GT, GE}:

            def callback(gmodel, where):
                if where == GRB.Callback.MIPNODE:
                    status = gmodel.cbGet(GRB.Callback.MIPNODE_STATUS)
                    if status == GRB.OPTIMAL:
                        obj = gmodel.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        if obj >= const:
                            print("Finished, property ok")
                            print("obj bound ", obj)
                            gmodel.terminate()
                if where == GRB.Callback.MIPSOL:
                    obj = gmodel.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < const:
                        print("Finished by finding a counter example.")

            return callback

    elif isinstance(output_formula, NAryDisjFormula) or isinstance(output_formula, NAryConjFormula):

        def callback(m, where):
            if where == GRB.Callback.MIPNODE:
                status = m.cbGet(GRB.Callback.MIPNODE_STATUS)
                if status == GRB.OPTIMAL:
                    obj = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
                    if obj < 0:
                        print("satisfied")
                        print("obj bound ", obj)
                        m.terminate()
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > 0:
                    print("Finished by failure = ")
                    m.terminate()

        return callback
    else:
        return None


def get_output_constrs(output_formula: Formula, model: Model, output_vars, negate: bool = True):
    if negate:
        negated_output_formula = NegationFormula(output_formula).to_nnf()
    else:
        negated_output_formula = output_formula
    # print(negated_output_formula)
    constrs = get_constrs(model=model, formula=negated_output_formula, vars=output_vars)
    model.update()
    for constr in constrs:
        model.addConstr(constr)
    model.update()
    return constrs


def get_constrs(model: Model, formula: Formula, vars: List[Var]):
    if isinstance(formula, Constraint):
        # Note: Constraint is a leaf (terminating) node.
        return [get_atomic_constr(formula, vars)]

    if isinstance(formula, ConjFormula):
        return get_constrs(model, formula.left, vars) + get_constrs(model, formula.right, vars)

    if isinstance(formula, NAryConjFormula):
        constrs = []
        for subformula in formula.clauses:
            constrs += get_constrs(model, subformula, vars)

        return constrs

    if isinstance(formula, DisjFormula):
        split_var = model.addVar(vtype=GRB.BINARY)
        clause_vars = [
            model.addVars(len(vars), lb=-GRB.INFINITY),
            model.addVars(len(vars), lb=-GRB.INFINITY),
        ]

        constr_sets = [
            get_constrs(model, formula.left, clause_vars[0]),
            get_constrs(model, formula.right, clause_vars[1]),
        ]

        constrs = []
        for i in [0, 1]:
            for j in range(len(vars)):
                constrs.append((split_var == i) >> (vars[j] == clause_vars[i][j]))

            for disj_constr in constr_sets[i]:
                constrs.append((split_var == i) >> disj_constr)

        return constrs

    if isinstance(formula, NAryDisjFormula):
        clauses = formula.clauses
        split_vars = model.addVars(len(clauses), vtype=GRB.BINARY)
        clause_vars = [model.addVars(len(vars), lb=-GRB.INFINITY) for _ in range(len(clauses))]

        constr_sets = []
        constrs = []
        for i in range(len(clauses)):
            constr_sets.append(get_constrs(model, clauses[i], clause_vars[i]))

            for j in range(len(vars)):
                constrs.append((split_vars[i] == 1) >> (vars[j] == clause_vars[i][j]))

            for disj_constr in constr_sets[i]:
                constrs.append((split_vars[i] == 1) >> disj_constr)

        # exactly one variable must be true
        constrs.append(quicksum(split_vars) == 1)

        return constrs

    raise Exception("unexpected formula", formula)


def get_atomic_constr(constraint: Constraint, vars: List[Var]):
    sense = constraint.sense
    offset = 0
    if isinstance(constraint, VarVarConstraint):
        op1 = vars[constraint.op1.i]
        op2 = vars[constraint.op2.i]
        offset = constraint.offset
    elif isinstance(constraint, VarConstConstraint):
        op1 = vars[constraint.op1.i]
        op2 = constraint.op2
    elif isinstance(constraint, LinExprConstraint):
        op1 = 0
        for i, c in constraint.op1.coord_coeff_map.items():
            op1 += c * vars[i]
        op2 = constraint.op2
    else:
        raise Exception("Unexpected type of atomic constraint", constraint)

    if sense == GE:
        return op1 >= op2 + offset
    elif sense == LE:
        return op1 <= op2 + offset
    elif sense == EQ:
        return op1 == op2 + offset
    else:
        raise Exception("Unexpected type of sense", sense)
