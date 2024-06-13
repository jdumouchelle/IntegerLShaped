# standard imports
import os
import time
import argparse
import numpy as np
import pickle as pkl
import gurobipy as gp
from pathlib import Path
import multiprocessing as mp

# integer L-shaped imports
from ils.two_sp import factory_two_sp
from ils.ils import factory_ils
from ils.utils import factory_get_path


#---------------------------------------------------------#
#                                                         #  
#         File to run the integer L-shaped method         #  
#                                                         #   
#---------------------------------------------------------#

"""
    Note that in order to achieve the best performance with multiprocessing
    most of the computations done in parallel are directly in this file rather than being
    in the integer L-shaped class.  This includes lower bound computation, benders,
    and the integer L-shaped method.
"""

#--------------------------------------------------------#
#           Functions to compute lower bound             #
#--------------------------------------------------------#

def compute_lb_worker(x, p_s, scen_idx, as_lp):
    """ Worker to compute lower bound for a particular second-stage problem. 

        Parameters
        -------------------------------------------------
            x: the first-stage decision
            p_s: a float of the scenario probability
            scen_idx:  the index of the scenario
            as_lp: an indicator for getting subproblems with or without the LP relaxation.
    """
    if as_lp:
        Q_s =  main_problem._lp_subproblems[scen_idx]
    else:
        Q_s =  main_problem._ip_subproblems[scen_idx]
        
    Q_s = ils.fix_first_stage_sol(Q_s, x)
    Q_s.optimize()
    
    return  p_s * Q_s.objVal


def compute_lower_bound(as_lp, lb_adjustment):
    """ Computes lower-bound for problem.

        Parameters
        -------------------------------------------------
            as_lp: an indicator for getting subproblems with or without the LP relaxation.
            lb_adjustment - a float to slightly lower the value to avoid any numerical issues.
    """
    x = ils.get_lower_bound_first_stage()
    
    # initialze pool if multiprocessing is used.
    # if ils.use_mp:
    #     pool = mp.Pool(ils.n_procs)

    results = []
    for scen_idx in range(ils.n_scenarios):
        # get relavent constraint/coefficient info
        p_s = ils.get_scenario_prob(scen_idx)

        # compute lower bound for subproblem
        if ils.use_mp:
            results.append(pool.apply_async(compute_lb_worker, (x, p_s, scen_idx, as_lp)))
        else:
            results.append(compute_lb_worker(x, p_s, scen_idx, as_lp))

    if ils.use_mp:
        results = [r.get() for r in results]

    lower_bound = np.sum(results)
    lower_bound -= lb_adjustment # adjust to avoid numerical issues

    return lower_bound




#--------------------------------------------------------#
#         Functions to Run Benders Decomposition         #
#--------------------------------------------------------#

def get_sp_cut_worker(x, p_s, h_s, T_s, scen_idx):
        """ Multiprocessing function for get sp cuts.
            Used in benders as well as the relaxed cuts in integer L-shaped method. 

            Parameters
            -------------------------------------------------
                x: the first-stage decision
                p_s: a float of the scenario probability
                h_s: a list or array of the rhs for a scenario
                T_s: a (nested) list or array of the linking constraints for a scenario.
                scen_idx:  the index of the scenario
                as_lp: an indicator for getting subproblems with or without the LP relaxation.
        """
        # fix first-stage variables
        fix_time = time.time()
        Q_s = main_problem._lp_subproblems[scen_idx]
        Q_s = ils.fix_first_stage_sol(Q_s, x)
        Q_s.setParam("Threads", ils.sp_threads)
        fix_time = time.time() - fix_time
        
        # optimize
        opt_time = time.time()
        Q_s.optimize()
        opt_time = time.time() - opt_time
        
        # get weighted objective
        q_val = p_s * Q_s.objVal
        
        # recover dual value
        dual_time = time.time()
        pi = ils.get_pi_vals(Q_s)
        dual_time = time.time() - dual_time

        # compute alpha/beta for cuts
        cut_time = time.time()
        alpha, beta = ils.compute_alpha_and_beta(pi, p_s, h_s, T_s)
        cut_time = time.time() - cut_time
        
        info = {
            "fix_time" : fix_time,
            "opt_time" : opt_time,
            "dual_time" : dual_time,
            "cut_time" : cut_time,
        }

        # check that values are correct
        assert(np.abs(alpha - np.dot(beta, x) - q_val) < 1e-2)
        
        return q_val, alpha, beta, info


def benders():
    """ Benders decompostion for linear relaxation of master problem. """
    time_benders = time.time()
    
    for it in range(ils.benders_max_iter):
        
        if (it%10) == 0:
            print(f"    Iteration: {it}")
        
        # optimize main problem
        main_problem.optimize()
    
        # recover solution
        x = list(map(lambda x: x.x, main_problem._x.values()))
        theta = main_problem._theta.x
        
        # solve all subproblems and recover duals in main problem.
        sp_time = time.time()

        results = []

        for scen_idx in range(ils.n_scenarios):
            # get relavent constraint/coefficient info
            p_s = ils.get_scenario_prob(scen_idx)
            T_s = ils.get_T_s(scen_idx)
            h_s = ils.get_h_s(scen_idx)

            # compute subgradient cut info
            if ils.use_mp:
                results.append(pool.apply_async(get_sp_cut_worker, (x, p_s, h_s, T_s, scen_idx)))
            else:
                results.append(get_sp_cut_worker(x, p_s, h_s, T_s, scen_idx))

        # if using multiprocessing, get from mp object
        if ils.use_mp:
            results = [r.get() for r in results]

        sp_time = time.time() - sp_time

        # get dual values
        cut_time = time.time()

        Q = np.sum(list(map(lambda x: x[0], results)))
        a_sum = np.sum(list(map(lambda x: x[1], results)))
        b_sum = np.sum(list(map(lambda x: x[2], results)), axis=0)

        cut_time = time.time() - cut_time

        if theta >= Q - ils.benders_tol:
            print("    Done: theta >= Q")
            break
    
        # otherwise add cut
        x_beta_sum_ = 0
        for i, x_var in main_problem._x.items():
            x_beta_sum_ += b_sum[i] * x_var

        main_problem.addConstr(main_problem._theta >= a_sum - x_beta_sum_, name=f"bc_{main_problem._bd_cuts}")
        main_problem._bd_cuts += 1
        
    time_benders = time.time() - time_benders
    main_problem._time_benders = time_benders
            


#-------------------------------------------------------------------------------#
#                         Integer L-shaped specific functions                   #
#-------------------------------------------------------------------------------#

def get_optimality_cut_worker(x, p_s, scen_idx):
    """ Multiprocessing function for get sp cut.             

        Parameters
        -------------------------------------------------
            x: the first-stage decision
            p_s: a float of the scenario probability
            scen_idx:  the index of the scenario
    """        
    # load subproblem, fix x, then optimize
    Q_s = main_problem._ip_subproblems[scen_idx]
    Q_s = ils.fix_first_stage_sol(Q_s, x)
    Q_s.optimize()
    
    q_val = p_s * Q_s.objVal
    
    return q_val


def compute_subgradient_cut(x, theta):
        """ Computes subgradient cuts based on IP solution of model. 

            Parameters
            -------------------------------------------------
                x: the first-stage decision in callback
                theta: the variable theta in callback
        """
        # x to list of values
        x = list(map(lambda y: y, x.values()))
        x = ils.round_x(x)
        
        # solve all subproblems and recover duals.    
        sp_time = time.time()    
        results = []

        for scen_idx in range(ils.n_scenarios):
            # get relavent constraint/coefficient info
            p_s = ils.get_scenario_prob(scen_idx)
            T_s = ils.get_T_s(scen_idx)
            h_s = ils.get_h_s(scen_idx)

            # compute sp cut info
            if ils.use_mp:
                results.append(pool.apply_async(get_sp_cut_worker, (x, p_s, h_s, T_s, scen_idx)))
            else:
                print(f"     Subgradient cut for scen {scen_idx}/{ils.n_scenarios}")
                results.append(get_sp_cut_worker(x, p_s, h_s, T_s, scen_idx))

        # get from mp is using
        if ils.use_mp:
            results = [r.get() for r in results]

        sp_time = time.time() - sp_time
        
        Q = np.sum(list(map(lambda x: x[0], results)))
        alpha = np.sum(list(map(lambda x: x[1], results)))
        beta = np.sum(list(map(lambda x: x[2], results)), axis=0)
        
        return Q, alpha, beta


def compute_integer_optimality_cut(x, theta):
    """ Integer optimality cut. 

        Parameters
        -------------------------------------------------
            x: the first-stage decision in callback
            theta: the variable theta in callback
    """
    # x to list of values
    x = list(map(lambda y: y, x.values()))
    x = ils.round_x(x)
    
    sp_time = time.time()
    results = []

    for scen_idx in range(main_problem._n_scenarios):
        # get relavent constraint/coefficient info
        p_s = ils.get_scenario_prob(scen_idx)

        if ils.use_mp:
            results.append(pool.apply_async(get_optimality_cut_worker, (x, p_s, scen_idx)))
        else:
            # print(f"     Optimality cut for scen {scen_idx}/{ils.n_scenarios}")
            results.append(get_optimality_cut_worker(x, p_s, scen_idx))

    if ils.use_mp:
        results = [r.get() for r in results]
    
    sp_time = time.time() - sp_time
    
    Q = np.sum(results)

    return Q



#-------------------------------------------------------------------------------#
#          Callback for Integer L-shaped method with alternating cuts           #
#-------------------------------------------------------------------------------#

def ils_alternating_cuts(main_problem, where):
    """ Callback function for integer L-shaped method with alternating cuts.  """        
    if where == gp.GRB.Callback.MIPSOL:

        # get number of nodes
        n_nodes = main_problem.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
        if n_nodes % 10 == 0:
            print("  # Nodes:", n_nodes)
        
        # get first-stage and theta
        x = main_problem.cbGetSolution(main_problem._x)
        theta = main_problem.cbGetSolution(main_problem._theta)

        # condition for satisfied integer solution
        if ils.hash_x(x) in main_problem._V:
            print("  Solution (x) in V, ending callback")
            return
        
        # if no sg cuts, then add them without adding integer cut
        if ils.hash_x(x) not in main_problem._V_lp: 
            print("  Solution (x) not in V_lp...")
            
            # compute Q_lp, and subgradient cut info
            Q_lp, alpha, beta = compute_subgradient_cut(x, theta)
            
            # add x to V_LP
            main_problem._V_lp.add(ils.hash_x(x))
            
            # add sg cut and return if theta < Q_lp
            if theta < Q_lp:
                print("  Adding subgradient cut (theta < Q_lp), ending callback")
                main_problem = ils.add_subgradient_cut(main_problem, alpha, beta)
                return
            
            print("  No subgradient cut needed (theta >= Q_lp)")
        
        # integer optimality cut Q value
        Q_ip = compute_integer_optimality_cut(x, theta)
        
        # if sg cuts, then add integer cuts
        main_problem._V.add(ils.hash_x(x))
        
        if theta < Q_ip:
            print("  Adding optimality (theta < Q_ip), ending callback")
            main_problem = ils.add_integer_optimality_cut(main_problem, x, Q_ip)
            return
        
        print("Did not meet any conditions!")



#-------------------------------------------------------------------------------#
#        Callback for Integer L-shaped method without alternating cuts          #
#-------------------------------------------------------------------------------#
def ils_standard(main_problem, where):
    raise Exception("To be implemented")




#-------------------------------------------------------------------------------#
#                                       args                                    #
#-------------------------------------------------------------------------------#

def get_args():
    """ Parser args for the script. """
    parser = argparse.ArgumentParser(description='Runs Integer-L Shaped method.')

    # problem args
    parser.add_argument('--problem', type=str, default="cflp_10_10")
    parser.add_argument('--n_scenarios', type=int, default=100)
    parser.add_argument('--test_set', type=int, default=0,  help='Evaluate on a test set.  For SSLP, 0 implies the SIPLIB scenarios')

    # algorithm to run
    #  - ils_ac: integer L-shaped method with alternating cuts (default, fastest option from [REF])
    #  - ils_std: integer L-shaped method with only integer cuts (not yet implemented)
    parser.add_argument('--algorithm', type=str, default='ils_ac', choices=['ils_ac', 'ils_std'])

    # data args
    parser.add_argument('--data_path', type=str, default="./data/",  help='Path to data.')

    # parallelism args (set to at most (n-1)-cores)
    parser.add_argument('--n_procs', type=int, default=4,  help='Number of procs for parallel subproblem solving.')

    # lower bound
    parser.add_argument('--lower_bound_adjustment', type=int, default=100, help='Decreasing lower bound by this amount.\
         In some cases, convergence to the results reported in Angulo et. at were not consistent with lower values.')

    # optimization parameters
    parser.add_argument('--benders_tol', type=int, default=1e-6, help='Optimality tolerance for benders decomposition.')
    parser.add_argument('--benders_max_iter', type=int, default=1000, help='Maximum number of iterations for benders.')

    # optimization parameters
    parser.add_argument('--mp_time_limit', type=int, default=3 * 3600, help='Time limit for main problem.')
    parser.add_argument('--mp_mipgap', type=float, default=1e-6, help='Gap limit for main problem.')
    parser.add_argument('--mp_threads', type=int, default=1, help='Number of threads for main problem.')

    # optimization parameters
    parser.add_argument('--sp_time_limit', type=int, default=3 * 3600, help='Time limit for subproblems.')
    parser.add_argument('--sp_mipgap', type=float, default=1e-6, help='Gap limit for subproblems.')
    parser.add_argument('--sp_threads', type=int, default=1, help='Number of threads for subproblems (should generally always be 1)).')
    
    args = parser.parse_args()

    return args




#-------------------------------------------------------------------------------#
#                                       main                                    #
#-------------------------------------------------------------------------------#

if __name__ == "__main__":

    # get args
    args = get_args()

    # load all two_sp specific information
    get_path = factory_get_path(args.problem)
    fp_inst = get_path(args.data_path, args.problem, ptype="inst", suffix=".pkl")

    with open(fp_inst, 'rb') as p:
        inst = pkl.load(p)

    # get two_sp and scenarios
    two_sp = factory_two_sp(args.problem, inst)
    scenarios = two_sp.get_scenarios(args.n_scenarios, args.test_set)

    # initialize integer L-shaped method
    global ils
    ils = factory_ils(args.problem, two_sp, scenarios)

    # set optimization params
    ils.set_opt_params(
        args.mp_threads, 
        args.mp_mipgap, 
        args.sp_threads, 
        args.sp_mipgap, 
        args.n_procs,
        args.benders_tol,
        args.benders_max_iter)

    # initialize main problem
    global main_problem
    main_problem, x, theta = ils.get_main_lp()

    # set x and theta in main problem
    main_problem._x = x
    main_problem._theta = theta

    # store additional useful info related to solving
    main_problem._use_mp = ils.use_mp
    main_problem._two_sp = two_sp
    main_problem._n_scenarios = ils.n_scenarios

    # set model parameters specified by args
    main_problem.setParam("MipGap", args.mp_mipgap)
    main_problem.setParam("Threads", args.mp_threads)
    main_problem.setParam("OutputFlag", 0)

    # get linear subproblems
    lp_subproblems = ils.get_subproblems(as_lp=True)
    main_problem._lp_subproblems = lp_subproblems

    # get integer subproblems
    ip_subproblems = ils.get_subproblems(as_lp=False)
    main_problem._ip_subproblems = ip_subproblems

    # add second-stage info
    ils.get_second_stage_info()

    # compute lower bound
    print("\nComputing Lower Bound...")
    time_lower_bound = time.time()

    # initialize mp pool for lower bound
    if ils.use_mp:
        global pool
        pool = mp.Pool(processes=args.n_procs)

    lower_bound = compute_lower_bound(as_lp=False, lb_adjustment=args.lower_bound_adjustment)

    if ils.use_mp:
        pool.close()
        pool.join()

    # set lower bound
    ils.set_lower_bound(main_problem, lower_bound)

    time_lower_bound = time.time() - time_lower_bound

    print("  Lower bound compute:", lower_bound)
    print("  Time for lower bound:", time_lower_bound)

    # benders
    print("\nBenders Decomposition...")

    # info for integer-L shaped with alternating cuts
    main_problem._bd_cuts = 0
    main_problem._sg_cuts = 0
    main_problem._io_cuts = 0
    main_problem.Params.lazyConstraints = 1

    main_problem._V = set()
    main_problem._V_lp = set()

    # run benders on continous relaxation
    if ils.use_mp:
        pool = mp.Pool(processes=args.n_procs)

    benders()

    benders_obj = main_problem.objVal

    print("  Objective of LP Relaxation:                ", benders_obj)
    print("  Solving time for Benders (LP Relaxation):  ", main_problem._time_benders)

    # integer L-shaped method
    print("\nInteger L-shaped Method...")
    main_problem = ils.set_first_stage_binary(main_problem)

    # optimize with callback
    if args.algorithm == "ils_ac":
        main_problem.optimize(ils_alternating_cuts)
    elif args.algorithm == "std":
        main_problem.optimize(ils_standard)

    print("\n\nSummary:")
    print("  Bender's cuts:", main_problem._bd_cuts)
    print("  Subgradient cuts:", main_problem._sg_cuts)
    print("  Integer cuts:", main_problem._io_cuts)

    print("  Number of Nodes:", main_problem.NodeCount)

    print("  Benders time:", main_problem._time_benders)
    print("  Time ILS:", main_problem.RunTime)
    print("  Time total:", main_problem._time_benders + main_problem.RunTime)

    # get first-stage decision
    x = main_problem._x.select()
    x = list(map(lambda y: y.x, x))

    # get objective from first-stage decision
    fs_obj = two_sp.evaluate_first_stage_sol(x, args.n_scenarios, test_set=args.test_set, n_procs=args.n_procs)

    print("\n  First-stage decision:", x)
    print("  First-stage solution obj:", fs_obj)

    # collect and store all results
    results = {
        'time_lower_bound' : time_lower_bound,
        'time_benders' : main_problem._time_benders,
        'time_integer_l_shaped' : main_problem.RunTime,
        'time_total' : time_lower_bound + main_problem._time_benders + main_problem.RunTime,

        'obj_benders' : benders_obj,
        'obj_fs' : fs_obj,

        'cuts_benders' : main_problem._bd_cuts,
        'cuts_subgradient' : main_problem._sg_cuts,
        'cuts_integer_opt' : main_problem._io_cuts,
        
        'n_nodes' : main_problem.NodeCount,

        'x' : x,
    }

    problem_str = f"s{args.n_scenarios}_ts{args.test_set}"
    fp_results = get_path(args.data_path, args.problem, ptype=f"ils_{problem_str}", suffix=".pkl")

    with open(fp_results, 'wb') as p:
        pkl.dump(results, p)
