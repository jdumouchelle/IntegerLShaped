# standard imports
import argparse

import os
import time
import pickle as pkl
import numpy as np
import gurobipy as gp
from pathlib import Path
import multiprocessing as mp
import matplotlib.pyplot as plt

# ils imports
from ils.two_sp import factory_two_sp
from ils.utils import factory_get_path

from .ils import IntegerLShaped


class CFLPIntegerLShaped(IntegerLShaped):


    #-------------------------------------------#
    #                Constructor                #
    #-------------------------------------------#

    def __init__(self, two_sp, scenarios):
        """ Constructor for CFLP Integer L-shaped method. """
        super(CFLPIntegerLShaped, self).__init__(two_sp, scenarios)

        self.n_facilities = self.two_sp.inst['n_facilities']
        self.n_customers = self.two_sp.inst['n_facilities']


    #-------------------------------------------#
    #          Problem-specific  methods        #
    #-------------------------------------------#

    def get_main_lp(self):
        """ Initialize model with first-stage variables and constraints. """
        main_problem = gp.Model()

        # variables
        x = main_problem.addVars(self.n_facilities, vtype="C", lb=0, ub=1, name='x', obj=self.two_sp.inst['fixed_costs'])    
        theta = main_problem.addVar(name="theta", vtype="C", lb=-gp.GRB.INFINITY, obj=1)
        
        return main_problem, x, theta


    def get_subproblems(self, as_lp):
        """ Gets subproblems.  """
        subproblems = []
        for i, scenario in enumerate(self.scenarios):
            Q_s = self.two_sp.make_second_stage_model(scenario)
            Q_s = self.fix_second_stage_model(Q_s, as_lp)
            
            Q_s.setParam("OutputFlag", 0)
            Q_s.setParam("Threads", self.sp_threads)
            Q_s.setParam("MipGap", self.sp_mipgap)
            
            Q_s._x = Q_s.getVars()[:self.n_facilities]
            Q_s._pi_constrs = Q_s.getConstrs()
            
            subproblems.append(Q_s)

        return subproblems


    def fix_second_stage_model(self, Q_s, as_lp):
        """  """
        # remove x from obj
        for i in range(self.n_facilities):
            var = Q_s.getVarByName(f"x_{i}")
            var.obj = 0
            
        Q_s.update()
        
        if as_lp:
            # relax mip to LP
            Q_s = Q_s.relax()
            
            # add constraints for upper bound of y values
            for var in Q_s.getVars():
                if "y_" in var.varName:
                    var.ub = gp.GRB.INFINITY
                    Q_s.addConstr(var + 0 <= 1, name=f"{var.varName}_ub")  
                if "z_" in var.varName:
                    var.ub = gp.GRB.INFINITY
                    Q_s.addConstr(var + 0 <= 1, name=f"{var.varName}_ub")  
                
        Q_s.update()
        
        return Q_s


    def get_second_stage_info(self):
        """ Computes and adds add second-stage info to main_problem. """        
        # number of constraints
        n_constrs_T = self.n_customers                          # constraints for meeting customer demand
        n_constrs_T += self.n_facilities                        # constriants for meet location capacity
        n_constrs_T += self.n_facilities * self.n_customers     # constraints for bound tightening
        n_constrs_T += self.n_facilities * self.n_customers     # constraints for upper bound of second-stage allocation vars
        n_constrs_T +=  self.n_customers                        # constraints for upper bound of second-stage recourse vars
        
        # constraint matrix for linking constraints
        T_s = np.zeros((n_constrs_T, self.n_facilities))
        
        # coefficients for demand constraints
        for i in range(self.n_facilities):
            T_s[i+self.n_customers,i] = self.two_sp.inst["capacities"][i]
        
        # coefficients for bound tightening constraints
        bc_start_idx = self.n_facilities + self.n_customers
        for i in range(self.n_facilities):
            s_idx = bc_start_idx + i * self.n_customers
            e_idx = bc_start_idx + (i+1) * self.n_customers
            T_s[s_idx:e_idx, i] = 1
                
        # right-hand side
        h = []
        for scenario in self.scenarios:
            h.append(self.get_rhs_vals(scenario))
        
        self.p_s = 1 / self.n_scenarios
        self.T_s = T_s
        self.h = h


    def get_rhs_vals(self, scenario):
        """ Gets h_s (i.e. RHS for second-stage problem) as a list.  """
        h = [-1] * self.n_customers                                 # RHS for client constraint
        h += [0] * self.n_facilities                                # RHS for demand constraint 
        h += [0] * self.n_facilities * self.n_customers             # RHS for bound tightening
        h += [-1] * self.n_facilities * self.n_customers            # RHS for y upper bound
        h += [-1] * self.n_customers                                # RHS for z upper bound
        h = np.array(h)
        return h


    def compute_alpha_and_beta(self, pi, p_s, h_s, T_s):
        """ Compute alpha and beta used in subgradient cuts. """
        alpha = - np.multiply(p_s, np.dot(pi, h_s))
        beta = - np.multiply(p_s, np.matmul(pi, T_s))
        return alpha, beta


    def get_lower_bound_first_stage(self):
        """ Gets best first-stage decision based on lower bound.  
            For CFLP this is opening all facilities. 
        """
        return [1] * self.n_facilities


    def get_scenario_prob(self, scen_idx):
        """ Gets scenario probabilities. """
        return self.p_s


    def get_T_s(self, scen_idx):
        """ Gets T matrix for a scen with index scen_idx. """
        return self.T_s


    def get_h_s(self, scen_idx):
        """ Gets h (rhs) for a scen with index scen_idx. """
        return self.h[scen_idx]
