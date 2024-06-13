from multiprocessing import Pool, Manager

import gurobipy as gp
import numpy as np

from .two_sp import TwoStageStocProg


class SSLP(TwoStageStocProg):

    def __init__(self, inst):
        """ Constructor for SSLP 2SP class. """
        super(SSLP, self).__init__(inst)


    def make_second_stage_model(self, scenario):
        """ Initializes a second stage model. """
        model = gp.Model()

        # variables
        x_vars, y_vars, r_vars = {}, {}, {}

        for loc in range(self.inst['n_locations']):
            v_name = f"x_{loc + 1}"
            obj = self.inst['first_stage_costs'][loc]
            x_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for clnt in range(self.inst['n_clients']):
            for loc in range(self.inst['n_locations']):
                v_name = f"y_{clnt + 1}_{loc + 1}"
                obj = self.inst['second_stage_costs'][clnt][loc]
                y_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for loc in range(self.inst['n_locations']):
            v_name = f"r_{loc + 1}"
            obj = self.inst['recourse_costs'][loc]
            r_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="C")

        # constraints
        constrs = {}

        # location limit constraints
        eq_ = 0
        for loc in range(self.inst['n_locations']):
            eq_ += x_vars[f"x_{loc + 1}"]

        c_name = "location_limit"
        constrs[c_name] = model.addConstr(eq_ <= self.inst['location_limit'], name=c_name)

        # location capacity constraints
        for loc in range(self.inst['n_locations']):
            eq_ = self.inst['location_coeffs'][loc] * x_vars[f"x_{loc + 1}"]
            eq_ += self.inst['recourse_coeffs'][loc] * r_vars[f"r_{loc + 1}"]
            for clnt in range(self.inst['n_clients']):
                eq_ += self.inst['client_coeffs'][clnt][loc] * y_vars[f"y_{clnt + 1}_{loc + 1}"]
            c_name = f"capacity_{loc + 1}"
            constrs[c_name] = model.addConstr(eq_ >= 0, name=c_name)

        # client constrints
        for clnt in range(self.inst['n_clients']):
            eq_ = 0
            for loc in range(self.inst['n_locations']):
                eq_ += y_vars[f"y_{clnt + 1}_{loc + 1}"]
            c_name = f'active_{clnt + 1}'
            constrs[c_name] = model.addConstr(eq_ == scenario[clnt], name=c_name)

        model.update()

        return model


    def get_second_stage_objective(self, x, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Gets the second stage model for an objective. """
        model = self.make_second_stage_model(scenario)
        model = self.set_first_stage(model, x)

        # optimize model
        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam('Threads', threads)
        model.optimize()

        return self.get_second_stage_cost(model)


    def evaluate_first_stage_sol(self, x, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0, test_set=0, n_procs=1):
        """ Gets the objective function value for a given solution. """
        scenarios = self.get_scenarios(n_scenarios, test_set)
        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios

        # get first stage objective
        fs_obj = np.dot(x, self.inst['first_stage_costs'])

        # second stage objective
        pool = Pool(n_procs)
        results = [pool.apply_async(self.get_second_stage_obj_worker, args=(x, scenario, scenario_prob, gap, time_limit, threads, verbose)) for scenario in scenarios]
        results = [r.get() for r in results]

        second_stage_obj_val = np.sum(results)

        return fs_obj + second_stage_obj_val



    def get_scenarios(self, n_scenarios, test_set):
        """ Gets n_scenario sceanrios.  Randomization based on test_set. """
        if test_set == 0:
            sslp_instance = f'sslp_{self.inst["n_locations"]}_{self.inst["n_clients"]}_{n_scenarios}'
            scenarios = self.inst['siplib_scenario_dict'][sslp_instance]

        else:
            # test_set = int(test_set)
            rng = np.random.RandomState()
            rng.seed(n_scenarios + test_set)

            scenarios = rng.randint(0, 2, size=(n_scenarios, self.inst['n_clients'])).tolist()

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


    def get_second_stage_obj_worker(self, x, scenario, scenario_prob, gap, time_limit, threads, verbose):
        """ Multiprocessing for getting second-stage objective. """
        second_stage_obj = scenario_prob * self.get_second_stage_objective(x, scenario, gap=gap, time_limit=time_limit, threads=threads, verbose=verbose)
        return second_stage_obj


    def get_second_stage_cost(self, model):
        """ Gets the second stage cost of a given model.  """
        ss_obj = 0
        for var in model.getVars():
            if "y" in var.varName or "r" in var.varName:
                ss_obj += var.obj * var.x
        return ss_obj


