from abc import ABC, abstractmethod


class TwoStageStocProg(ABC):
    """
    Class for Two staged Stochastic Integer Programming Problem.  This should store and be able to construct all
    useful problems/subproblems (as Gurobi models) as well as evaluate solutions, and sample scenarios.
    """


    #-------------------------------------------#
    #               Initialization              #
    #-------------------------------------------#

    def __init__(self, inst):
        """ Constructor for TwoStageStocProg class. 
            
            Parameters
            -------------------------------------------------
                inst: information related to the instance.  size, coefficients, etc.  See CFLP/SSLP for example.
        """
        self.inst = inst
 


    #-------------------------------------------#
    #                Abstract Methods           #
    #-------------------------------------------#

    @abstractmethod
    def make_second_stage_model(self, scenario):
        """ Creates a second stage problem for a given scenario.             

            Parameters
            -------------------------------------------------
                scenario: The scenario to create the model for. 
        """
        pass

    @abstractmethod
    def get_second_stage_objective(self, x, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Solves the second stage problem for a given scenario.             

            Parameters
            -------------------------------------------------
                x: the first-stage decision (as a list/array).
                scenario: the scenario.
                gap: MIPGap for Gurobi.
                time_limit: TimeLimit for Gurobi.
                threads: Number of threads for Gurobi (should generally be set to 1).
                verbose: Gurobi model verbosity. 
        """
        pass

    @abstractmethod
    def evaluate_first_stage_sol(self, x, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0, test_set=0, n_procs=1):
        """ Evaluates the first stage solution across all scenarios.             

            Parameters
            -------------------------------------------------
                x: the first-stage decision (as a list/array).
                n_scenarios: the number of scenarios.
                gap: MIPGap for Gurobi.
                time_limit: TimeLimit for Gurobi.
                threads: Number of threads for Gurobi (should generally be set to 1).
                verbose: Gurobi model verbosity. 
                test_set: index of test set. 
                n_procs: number of processes to use for solving in parallel.
        """
        pass

    @abstractmethod
    def get_scenarios(self, n_scenarios, test_set):
        """ Evaluates the first stage solution across all scenarios.        

            Parameters
            -------------------------------------------------
                n_scenarios: the number of scenarios.
                test_set: index of test set. 
        """
        pass

    @abstractmethod
    def set_first_stage(self, model, x):
        """ Fixes the first stage solution of a given model.      

            Parameters
            -------------------------------------------------
                model: the gurobi model to fix the decision in.
                x: the first-stage solution as a list/array.
        """
        pass
