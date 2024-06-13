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


class SSLPIntegerLShaped(IntegerLShaped):


    #-------------------------------------------#
    #                Constructor                #
    #-------------------------------------------#

    def __init__(self, two_sp, scenarios):
        """ Constructor for CFLP Integer L-shaped method. """
        super(SSLPIntegerLShaped, self).__init__(two_sp, scenarios)

        self.n_locations = two_sp.inst['n_locations']
        self.n_clients = two_sp.inst['n_clients']



    #-------------------------------------------#
    #          Problem-specific  methods        #
    #-------------------------------------------#

    def get_main_lp(self):
        """ Initialize model with first-stage variables and constraints. """
        main_problem = gp.Model()

        # Variables
        x = main_problem.addVars(self.n_locations, vtype="C", lb=0, ub=1, name='x', obj=self.two_sp.inst['first_stage_costs'])    
        theta = main_problem.addVar(name="theta", vtype="C", lb=-gp.GRB.INFINITY, obj=1)
                
        return main_problem, x, theta


    def get_subproblems(self, as_lp):
        """ Get list of subproblems.  """
        subproblems = []
        for i, scenario in enumerate(self.scenarios):
            Q_s = self.two_sp.make_second_stage_model(scenario)
            Q_s = self.fix_second_stage_model(Q_s, self.n_locations)

            Q_s.setParam("OutputFlag", 0)
            Q_s.setParam("Threads", self.sp_threads)
            Q_s.setParam("MipGap", self.sp_mipgap)

            Q_s._x = Q_s.getVars()[:self.n_locations]
            Q_s._pi_constrs = Q_s.getConstrs()
            
            subproblems.append(Q_s)

        return subproblems


    def fix_second_stage_model(self, Q_s, as_lp):
        """  """
        Q_s.remove(Q_s.getConstrByName("location_limit"))
        
        # remove x from obj
        for i in range(self.n_locations):
            var = Q_s.getVarByName(f"x_{i+1}")
            var.obj = 0
            
        Q_s.update()
        
        if as_lp:
            Q_s = Q_s.relax()
            
            # add constraints for upper bound of y values
            # needed to explicilty get dual values for variable upperbounds
            for var in Q_s.getVars():
                if "y_" in var.varName:
                    var.ub = gp.GRB.INFINITY
                    Q_s.addConstr(var + 0 <= 1, name=f"{var.varName}_ub")  
                
        Q_s.update()
        
        return Q_s


    def get_second_stage_info(self):
        """ Computes and adds add second-stage info to main_problem. """        
        T_s = np.zeros((self.n_locations, self.n_locations))
        for i in range(self.n_locations):
            T_s[i,i] = self.two_sp.inst["location_coeffs"][i]

        h = []
        for s in self.scenarios:
            h.append(self.get_rhs_vals(s))

        self.p_s = 1 / self.n_scenarios
        self.T_s = T_s
        self.h = h


    def get_rhs_vals(self, scenario):
        """ Gets h_s (i.e. RHS for second-stage problem) as a list. """
        h = [0] * self.n_locations 
        h += list(scenario) 
        h += [1] * (self.n_locations *  self.n_clients)
        h = np.array(h)
        return h

    def compute_alpha_and_beta(self, pi, p_s, h_s, T_s):
        """ Compute alpha and beta used in subgradient cuts. """
        alpha = np.multiply(p_s, np.dot(pi, h_s))
        beta = np.multiply(p_s, np.multiply(pi[:T_s.shape[0]], T_s[0,0]))
        return alpha, beta


    def get_lower_bound_first_stage(self):
        """ Gets best first-stage decision based on lower bound.  
            For SSLP this is opening all severs. 
        """
        return [1] * self.n_locations
        

    def get_scenario_prob(self, scen_idx):
        """ Gets scenario probabilities. """
        return self.p_s


    def get_T_s(self, scen_idx):
        """ Gets T matrix for a scen with index scen_idx. """
        return self.T_s


    def get_h_s(self, scen_idx):
        """ Gets h (rhs) for a scen with index scen_idx. """
        return self.h[scen_idx]
        
