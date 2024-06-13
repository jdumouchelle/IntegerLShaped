import time
import numpy as np
import gurobipy as gp
import multiprocessing as mp

from abc import ABC, abstractmethod


#-------------------------------------------------------------------------------#
#                      Class for Integer L-shaped subroutines                   #
#-------------------------------------------------------------------------------#

class IntegerLShaped(ABC):
    """
    Class for Integer L-shaped method.
        - This this class includes all of the methods to run the integer L-shaped method.
        - Some additional methods, such as constructing the subproblems will need to be implemented in gurobi 
          for each additional problem.
        - See CFLP/SSLP classes for examples.
    """

    #-------------------------------------------#
    #               Initialization              #
    #-------------------------------------------#

    def __init__(self, two_sp, scenarios):
        """ Constructor for Integer L-shaped method. 
            
            Parameters
            -------------------------------------------------
                two_sp: a TwoStageStocProg class (see ils.two_sp.two_sp)
                scenarios: a set of scenarios as a list.
        """
        self.two_sp = two_sp
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)



    #-------------------------------------------#
    #                Abstract Methods           #
    #-------------------------------------------#

    @abstractmethod
    def get_main_lp(self):
        """ Initialize model with first-stage variables and constraints. """
        pass

    @abstractmethod
    def get_subproblems(self, as_lp):
        """ Get list of subproblems.  This should be a list of gurobi models. 

            Parameters
            -------------------------------------------------
                as_lp: an indicator for getting subproblems with or without the LP relaxation.
        """
        pass

    @abstractmethod
    def fix_second_stage_model(self, Q_s, as_lp):
        """ Adds additional constraints/relaxations to the second-stage model.  
            This can include:
              - adding redundant constraints for variable upperbounds (required to get dual values)
              - relaxing and fixing variables
            See SSLP/CFLP for examples.  

            Parameters
            -------------------------------------------------
                Q_s: a gurobi model of the second-stage problem for a particular scenario.
                as_lp: an indicator for getting subproblems with or without the LP relaxation.
        """
        pass

    @abstractmethod
    def get_second_stage_info(self):
        """ Gets second-stage information store within class.  
            This should include info related to:
                - the scenario probabilities (p_s)
                - the linking constraints for each scenario (T_s)
                - the rhs constrints for each scenario (h_s).
            See CFLP/SSLP for examples.
        """        
        pass

    @abstractmethod
    def compute_alpha_and_beta(self, pi, p_s, h_s, T_s):
        """ Compute alpha and beta used in subgradient cuts. 
            See paper for more details on alpga/beta.  
              - Angulo, G., Ahmed, S., & Dey, S. S. (2016). Improving the integer L-shaped method. INFORMS Journal on Computing, 28(3), 483-499. 
              - https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2016.0695?journalCode=ijoc
            This will change slightly depending on the problem class, so see CFLP/SSLP for examples. 

            Parameters
            -------------------------------------------------
                pi: a list or array of the dual values for a scenario
                p_s: a float of the scenario probability
                h_s: a list or array of the rhs for a scenario
                T_s: a (nested) list or array of the linking constraints for a scenario.
        """
        pass

    @abstractmethod
    def get_lower_bound_first_stage(self):
        """ Gets best first-stage decision based on lower bound. 
            For CFLP/SSLP, this occurs with the maximum number of facilities/servers.
            This can also just be set to an arbitrarily low value.  """
        pass
        
    @abstractmethod
    def get_scenario_prob(self, scen_idx):
        """ Gets scenario probabilities for scenario at index scen_idx.

            Parameters
            -------------------------------------------------
                scen_idx: the index of the scenario
        """
        pass

    @abstractmethod
    def get_T_s(self, scen_idx):
        """ Gets T matrix for a scen with index scen_idx.  

            Parameters
            -------------------------------------------------
                scen_idx: the index of the scenario
        """
        pass

    @abstractmethod
    def get_h_s(self, scen_idx):
        """ Gets h (rhs) for a scen with index scen_idx. 

            Parameters
            -------------------------------------------------
                scen_idx: the index of the scenario
        """
        pass
        


    #--------------------------------------------------------#
    #               General Base Class Methods               #
    #--------------------------------------------------------#

    def set_opt_params(self, mp_threads, mp_mipgap, sp_threads, sp_mipgap, n_procs, benders_tol, benders_max_iter):
        """ A method to store all parameters used in optimization that will be routinely called.
            These will generally just be set in ils.scripts.run_ils.

            Parameters
            -------------------------------------------------
                mp_threads: Number of threads for main problem.
                mp_mipgap: Gap to use for main problem.
                sp_threads: Number of threads for subproblem. This should be set to 1 to utilize
                    more threads in parallel for each subproblem.
                sp_mipgap: Gap to use for subproblems.
                n_procs: Number of processes for parallelization.  
                benders_tol: tolerance for optimality in benders decomposition.
                benders_max_iter: maximum number of iterations for benders decomposition. 
        """
        # main problem
        self.mp_threads = mp_threads
        self.mp_mipgap = mp_mipgap

        # subproblem
        self.sp_threads = sp_threads
        self.sp_mipgap = sp_mipgap

        # number of processes
        self.n_procs = n_procs

        # set paralleism bool to True/False
        self.use_mp = True
        if self.n_procs == 1:
            self.use_mp = False

        self.benders_tol = benders_tol
        self.benders_max_iter = benders_max_iter


    def fix_first_stage_sol(self, Q_s, x):
        """ Fixes first-stage decision in second-stage model.
            
            Parameters
            -------------------------------------------------
                Q_s - A gurobi model of the second-stage problem
                x - The solution to set the first-stage decision to
        """    
        Q_s.setAttr(gp.GRB.Attr.LB, Q_s._x, x)
        Q_s.setAttr(gp.GRB.Attr.UB, Q_s._x, x)
        return Q_s


    def set_first_stage_binary(self, main_problem):
        """ Sets first-stage variables to binary.  Used after benders.  """
        for _, var in main_problem._x.items():
            var.vtype = "B"
        return main_problem


    def get_pi_vals(self, Q_s):
        """ Get dual values from second-stage model as numpy array. 

            Parameters
            -------------------------------------------------
                Q_s - A gurobi model of the second-stage problem
        """
        pi = Q_s.getAttr(gp.GRB.Attr.Pi, Q_s._pi_constrs)
        return pi


    def round_x(self, x):
        """ Rounds x to binary values.  Rounds values to avoid numerical issues.  

            Parameters
            -------------------------------------------------
                x - the first-stage decision (as a list).
        """
        x = list(map(lambda x: max(x, 0), x)) # fix value to be >= 0
        x = list(map(lambda x: min(x, 1), x)) # fix value to be <= 1
        return x


    def hash_x(self, x):
        """ Hashes binary first-stage solution to tuple.  Rounds values to avoid numerical issues.  

            Parameters
            -------------------------------------------------
                x - the first-stage decision (as a list).
        """
        xh = x.select()
        for i in range(len(xh)):
            if xh[i] < 0.5:
                xh[i] = 0
            else:
                xh[i] = 1
        return tuple(xh)


    #--------------------------------------------------------#
    #                       Lower Bounds                     #
    #--------------------------------------------------------#

    def set_lower_bound(self, main_problem, lower_bound):
        """ Sets lower-bound. 

            Parameters
            -------------------------------------------------
                lower_bound - the lower bound to set.
        """
        main_problem._lower_bound = lower_bound
        main_problem._theta.lb = lower_bound
        self.lower_bound = lower_bound
        return main_problem


    #--------------------------------------------------------#
    #           Methods for Integer L-shaped Method          #
    #--------------------------------------------------------#

    def add_subgradient_cut(self, main_problem, alpha, beta):
        """ Adds subgradient cut to master_problem. 

            Parameters
            -------------------------------------------------
                alpha: alpha for the subgradient cut
                beta: beta for the subgradient cut
        """
        x_beta_sum_ = 0
        for i, x_var in main_problem._x.items():
            x_beta_sum_ += beta[i] * x_var
        main_problem.cbLazy(main_problem._theta >= alpha - x_beta_sum_)
        main_problem._sg_cuts += 1
        return main_problem


    def add_integer_optimality_cut(self, main_problem, x, Q):
        """ Add integer optimality cut to main_problem. 
            
            Parameters
            -------------------------------------------------
                x: the first-stage decision in callback
                Q: the value for  the integer cut
        """
        x = list(map(lambda y: y, x.values()))
        x = self.round_x(x)

        # compute the set S for integer cut
        S = []
        S_not = []
        for i, var in main_problem._x.items():
            if x[i] > 0.99:
                S.append(var)
            else:
                S_not.append(var)
                
        # compute and add integer cut
        integer_cut = (Q - self.lower_bound) * (sum(S) - sum(S_not) - len(S)) + Q 
        main_problem.cbLazy(main_problem._theta >= integer_cut)
        main_problem._io_cuts += 1
        
        return main_problem


