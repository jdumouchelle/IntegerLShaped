from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from .two_sp import TwoStageStocProg


class CFLP(TwoStageStocProg):

    def __init__(self, inst):
        """ Constructor for CFLP 2SP class. """
        super(CFLP, self).__init__(inst)
        self.n_customers = self.inst['n_customers']
        self.n_facilities = self.inst['n_facilities']
        self.integer_second_stage = self.inst['integer_second_stage']
        self.bound_tightening_constrs = self.inst['bound_tightening_constrs']
        self.capacities = self.inst['capacities']
        self.fixed_costs = self.inst['fixed_costs']
        self.trans_costs = self.inst['trans_costs']
        self.recourse_costs = 2 * np.max([np.max(self.fixed_costs), np.max(self.trans_costs)])


    def make_second_stage_model(self, scenario):
        """ Creates the second stage model. """
        model = gp.Model()
        var_dict = {}

        # binary variables for each location
        for i in range(self.n_facilities):
            var_name = f"x_{i}"
            # bound lower and upper to solution
            var_dict[var_name] = model.addVar(obj=self.fixed_costs[i], vtype="B", name=var_name)

        # add either continous or binary second stage serving costs
        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                var_name = f"y_{i}_{j}"
                if self.integer_second_stage:
                    var_dict[var_name] = model.addVar(obj=self.trans_costs[i, j], vtype="B", name=var_name)
                else:
                    var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.trans_costs[i, j], vtype="C",
                                                      name=var_name)

        # add either continous or binary second stage recourse costs
        for j in range(self.n_customers):
            var_name = f"z_{j}"
            if self.integer_second_stage:
                var_dict[var_name] = model.addVar(obj=self.recourse_costs, vtype="B", name=var_name)
            else:
                var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.recourse_costs, vtype="C", name=var_name)

        model.update()

        # add demand constraints
        for j in range(self.n_customers):
            cons = var_dict[f"z_{j}"]
            for i in range(self.n_facilities):
                cons += var_dict[f"y_{i}_{j}"]
            model.addConstr(cons >= 1, name=f"d_{j}")

        # capacity constraints
        for i in range(self.n_facilities):
            cons = - self.capacities[i] * var_dict[f"x_{i}"]
            for j in range(self.n_customers):
                cons += scenario[j] * var_dict[f"y_{i}_{j}"]
            model.addConstr(cons <= 0, name=f"c_{i}")

        # bound tightening constraints
        if self.bound_tightening_constrs:
            for i in range(self.n_facilities):
                for j in range(self.n_customers):
                    model.addConstr(- var_dict[f"x_{i}"] + var_dict[f"y_{i}_{j}"] <= 0, name=f"t_{i}_{j}")

        model.update()

        return model


    def get_second_stage_objective(self, x, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Gets the second stage model for an objective. """
        model = self.make_second_stage_model(scenario)

        # fix first stage solution
        model = self.set_first_stage(model, x)

        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize()

        second_stage_obj = self.get_second_stage_cost(model)
        return second_stage_obj


    def evaluate_first_stage_sol(self, x, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0, test_set="0", n_procs=1):
        """ Gets the objective function value for a given solution. """
        scenarios = self.get_scenarios(n_scenarios, test_set)
        n_scenarios = len(scenarios)
        scenario_prob = 1 / len(scenarios)

        # get first-stage objective
        first_stage_obj_val = np.dot(x, self.fixed_costs)

        # get second-stage objective
        pool = Pool(n_procs)
        results = [pool.apply_async(self.get_second_stage_obj_worker, args=(x, scenario, scenario_prob, gap, time_limit, threads, verbose)) for scenario in scenarios]
        results = [r.get() for r in results]
        pool.close()
        pool.join()

        second_stage_obj_val = np.sum(results)

        return first_stage_obj_val + second_stage_obj_val


    def get_scenarios(self, n_scenarios, test_set):
        """ Gets n_scenario sceanrios.  Randomization based on test_set. """
        test_set = int(test_set)
        rng = np.random.RandomState()
        rng.seed(n_scenarios + test_set)
        scenarios = [] 
        for _ in range(n_scenarios):
            scenarios.append(rng.randint(5, 35 + 1, size=self.n_customers))

        return scenarios


    def set_first_stage(self, model, x):
        """ Fixes the first stage solution of a given model. """
        for var in model.getVars():
            if "x_" in var.varName:
                idx = int(var.varName.split("_")[-1])
                var.ub = x[idx-1]
                var.lb = x[idx-1]
        model.update()
        return model


    def set_first_stage(self, model, x):
        """ Fixes the first stage solution of a given model. """
        for var in model.getVars():
            if "x_" in var.varName:
                idx = int(var.varName.split("_")[-1])
                model.getVarByName(var.varName).lb = x[idx]
                model.getVarByName(var.varName).ub = x[idx]
        model.update()
        return model


    def get_second_stage_obj_worker(self, x, scenario, scenario_prob, gap, time_limit, threads, verbose):
        """ Multiprocessing for getting second-stage objective. """
        second_stage_obj = scenario_prob * self.get_second_stage_objective(x, scenario, gap=gap, time_limit=time_limit, verbose=verbose)
        return second_stage_obj


    def get_second_stage_cost(self, model):
        """ Gets the second stage cost of a given model.  """
        second_stage_obj = 0
        for var in model.getVars():
            if "x" not in var.varName:
                second_stage_obj += var.obj * var.x
        return second_stage_obj



