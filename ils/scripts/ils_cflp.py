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



## Functions for Initializing model

def get_master_lp(args, two_sp):
    """ Initialize model with first-stage variables and constraints. 
        Parameters:
            two_sp - a TwoStageStocProg instance
            args - arguments
    """
    master_problem = gp.Model()

    # Variables
    x = master_problem.addVars(two_sp.inst['n_facilities'], vtype="C", lb=0, ub=1, name='x', obj=two_sp.inst['fixed_costs'])    
    theta = master_problem.addVar(name="theta", vtype="C", lb=-gp.GRB.INFINITY, obj=1)
    
    # store variables and two_sp instance
    master_problem._x = x
    master_problem._theta = theta
    master_problem._two_sp = two_sp
    
    # set model parameters specified by args
    master_problem.setParam("MipGap", args.mipgap)
    master_problem.setParam("Threads", args.threads)
    master_problem.setParam("OutputFlag", 0)
    
    return master_problem



def fix_second_stage_model(Q_s, n_facilities, lp=True, add_ub=True):
    """ Modifies second-stage gurobi model with repesect to parameters.  Remove first-stage constraints
        and costs.
        Parameters:
            Q_s - A gurobi model of the second-stage problem
            n_facilities - Number of facilities (# of first-stage decisions)
            lp - Boolean indicating if problems should be LP (true) or IP (false)
            add_ub - Boolean indicating if upper bounds should be explicitly added as constraints 
                     required for recovering dual values efficiently
    """
    # remove first-stage constraints
    # Q_s.remove(Q_s.getConstrByName("location_limit"))
    
    # remove x from obj
    for i in range(n_facilities):
        var = Q_s.getVarByName(f"x_{i}")
        var.obj = 0
        
    Q_s.update()
    
    if lp:
        Q_s = Q_s.relax()
        
        # add constraints for upper bound of y values
        # needed to explicilty get dual values for variable upperbounds
        if add_ub:
            for var in Q_s.getVars():
                if "y_" in var.varName:
                    var.ub = gp.GRB.INFINITY
                    Q_s.addConstr(var + 0 <= 1, name=f"{var.varName}_ub")  
                if "z_" in var.varName:
                    var.ub = gp.GRB.INFINITY
                    Q_s.addConstr(var + 0 <= 1, name=f"{var.varName}_ub")  
            
    Q_s.update()
    
    return Q_s


def get_subproblems(args, two_sp, scenarios, lp=True, add_ub=True):
    """ Get list of subproblems for a given set of scenarios.
        Parameters:
            two_sp - a TwoStageStocProg instance
            scenarios - a list of scenarios
            n_facilities - Number of facilities (# of first-stage decisions)
            lp - Boolean indicating if problems should be LP (true) or IP (false)
            add_ub - Boolean indicating if upper bounds should be explicitly added as constraints 
                     required for recovering dual values efficiently
    """
    subproblems = []
    for i, s in enumerate(scenarios):
        Q_s = two_sp._make_second_stage_model(s)
        Q_s = fix_second_stage_model(Q_s, two_sp.inst["n_facilities"], lp, add_ub)
        
        Q_s.setParam("OutputFlag", 0)
        Q_s.setParam("Threads", args.threads)
        Q_s.setParam("MipGap", args.mipgap)
        
        Q_s._x = Q_s.getVars()[:two_sp.inst["n_facilities"]]
        Q_s._pi_constrs = Q_s.getConstrs()
        
        subproblems.append(Q_s)
    return subproblems


def fix_first_stage_sol(Q_s, sol):
    """ Fixes first-stage decision in second-stage model.
        Parameters:
            Q_s - A gurobi model of the second-stage problem
            sol - The solution to set the first-stage decision to
    """    
    Q_s.setAttr(gp.GRB.Attr.LB, Q_s._x, sol)
    Q_s.setAttr(gp.GRB.Attr.UB, Q_s._x, sol)
    return Q_s


def get_pi_vals(Q_s):
    """ Get dual values from second-stage model as numpy array. 
        Parameters:
            Q_s - A gurobi model of the second-stage problem
    """
    pi = Q_s.getAttr(gp.GRB.Attr.Pi, Q_s._pi_constrs)
    return pi


def get_h_vals(two_sp, scenario):
    """ Gets h_s (i.e. RHS for second-stage problem) as a list.
        Parameters:
            two_sp - a TwoStageStocProg instance
            scenario - a scenario
    """
    h = [-1] * two_sp.inst["n_customers"]                                    # RHS for client constraint
    #h += list(scenario)                                                    # RHS for demand constraint 
    h += [0] * two_sp.inst["n_facilities"]                                                       # RHS for demand constraint 
    h += [0] * (two_sp.inst["n_facilities"] * two_sp.inst["n_customers"])   # RHS for bound tightening
    h += [-1] * (two_sp.inst["n_facilities"] * two_sp.inst["n_customers"])   # RHS for y upper bound
    h += [-1] * two_sp.inst["n_customers"]                                   # RHS for z upper bound
    h = np.array(h)
    return h


def add_second_stage_info_to_mp(master_problem):
    """ Computes and adds add second-stage info to master_problem.
        Parameters:
            master_problem - A gurobi model of the master problem.
    """
    master_problem._p_s = 1 / master_problem._n_scenarios
    
    n_customers = two_sp.inst["n_customers"]
    n_facilities = two_sp.inst["n_facilities"]
    
    n_constrs_T = n_customers                   # constraints for meeting customer demand
    n_constrs_T += n_facilities                 # constriants for meet location capacity
    n_constrs_T += n_facilities * n_customers   # constraints for bound tightening
    n_constrs_T += n_facilities * n_customers   # constraints for upper bound of second-stage allocation vars
    n_constrs_T +=  n_customers                 # constraints for upper bound of second-stage recourse vars
    
    T_s = np.zeros((n_constrs_T, two_sp.inst["n_facilities"]))
    
    # coefficients for demand constraints
    for i in range(n_facilities):
        T_s[i+n_customers,i] = two_sp.inst["capacities"][i]
    
    # coefficients for bound tightening constraints
    bc_start_idx = n_facilities + n_customers
    for i in range(n_facilities):
        s_idx = bc_start_idx + i * n_customers
        e_idx = bc_start_idx + (i+1) * n_customers
        T_s[s_idx:e_idx, i] = 1
    
    master_problem._T_s = T_s
    
    h = []
    for s in scenarios:
        h.append(get_h_vals(two_sp, s))
    master_problem._h = h
    
    master_problem.update()
    return master_problem


###  Lower Bound Functions

def compute_lb_worker(x, p_s, scen_idx, lp):
    """ Worker to compute lower bound for a particular second-stage problem. 
        Parameters:
            x - First-stage solution
            p_s - Probability of scenario
            scen_idx - Index of scenario
            lp - Boolean indicating if problems should be LP (true) or IP (false)
    """
    if lp:
        Q_s =  master_problem._lp_subproblems[scen_idx]
    else:
        Q_s =  master_problem._ip_subproblems[scen_idx]
        
    Q_s = fix_first_stage_sol(Q_s, x)
    Q_s.optimize()
    
    return  p_s * Q_s.objVal


def compute_lower_bound(args, master_problem, two_sp, lp=False):
    """ Computes lower bound for the second-stage problem.  For SSLP, the lower
        bound is given when all facilities are open as the serving cost is minized.  
        Parameters:
            args -  args passed into argparse
            master_problem - The master problem
            two_sp - a TwoStageStocProg instance
            lp - Boolean indicating if problems should be LP (true) or IP (false)
    """
    x = [1] * two_sp.inst['n_facilities'] # best case if when all servers are open
    p_s = master_problem._p_s

    results = []
    
    if master_problem._use_mp:
        pool = mp.Pool(args.n_procs)
        
        for scen_idx in range(master_problem._n_scenarios):
            results.append(
                pool.apply_async(compute_lb_worker, (x, p_s, scen_idx, lp)))

        results = [r.get() for r in results]
    
    else:
        results = []
        for scen_idx in range(master_problem._n_scenarios):
            print(f"Computing LB for scen {scen_idx}/{master_problem._n_scenarios}")
            results.append(compute_lb_worker(x, p_s, scen_idx, lp))

    Q = np.sum(results)
        
    return Q


def set_lb(master_problem, L):
    """ Sets lower bound for theta. """
    theta = master_problem.getVarByName("theta")
    theta.lb = L
    master_problem._L = L


def get_sp_cut_worker(x, p_s, h_s, T_s, scen_idx):
    """ Multiprocessing function for get sp cut. """
    
    # fix first-stage variables
    fix_time = time.time()
    Q_s = master_problem._lp_subproblems[scen_idx]
    Q_s = fix_first_stage_sol(Q_s, x)
    fix_time = time.time() - fix_time
    
    # optimize
    opt_time = time.time()
    Q_s.optimize()
    opt_time = time.time() - opt_time
    
    # get weighted objective
    try:
        q_val = p_s * Q_s.objVal
    except:
        print("UNABLE TO GET OBJECTIVE")
        print("  x:", x)
        print("  i:", scen_idx)
        Q_s.write("error_problem.lp")
        x = "1" + 2
    
    # recover dual value
    dual_time = time.time()
    pi = get_pi_vals(Q_s)
    dual_time = time.time() - dual_time

    # compute alpha/beta for cuts
    cut_time = time.time()
    alpha = - np.multiply(p_s, np.dot(pi, h_s))
    beta = - np.multiply(p_s, np.matmul(pi, T_s))
    cut_time = time.time() - cut_time
    
    info = {
        "fix_time" : fix_time,
        "opt_time" : opt_time,
        "dual_time" : dual_time,
        "cut_time" : cut_time,
    }
    
    if np.abs(alpha - np.dot(beta, x) - q_val) > 1e-2:
        print(x)
        print(scen_idx)
        print("alpha:            ", alpha)
        print("beta @ x:         ", np.dot(beta, x))
        print("alpha - beta @ x: ", alpha - np.dot(beta, x))
        print("p_s * Q_s:        ", q_val)
        print("DIFF:             ", np.abs(alpha - np.dot(beta, x) - q_val))
        
    assert(np.abs(alpha - np.dot(beta, x) - q_val) < 1e-2)
    
    return q_val, alpha, beta, info


def benders(master_problem, tol=1e-3, n_max_iters=10000):
    
    # variables consistent between scenarios
    T_s = master_problem._T_s
    p_s = master_problem._p_s
    h = master_problem._h
    
    time_benders = time.time()
    
    for it in range(n_max_iters):
        
        print(f"Iteration: {it}")
        
        # optimize MP
        master_problem.optimize()
    
        # recover solution
        x = list(map(lambda x: x.x, master_problem._x.values()))
        x = list(map(lambda x: max(x, 0), x)) # fix value to be >= 0
        x = list(map(lambda x: min(x, 1), x)) # fix value to be <= 1

        theta = master_problem._theta.x
        
        # solve all subproblems and recover duals in MP.
        sp_time = time.time()

        results = []

        if master_problem._use_mp:
            for scen_idx in range(master_problem._n_scenarios):
                results.append(
                    pool.apply_async(get_sp_cut_worker, (x, p_s, h[scen_idx], T_s, scen_idx)))
            results = [r.get() for r in results]

        else:
            for scen_idx in range(master_problem._n_scenarios):
                print(f"     Benders for scen {scen_idx}/{master_problem._n_scenarios}")
                results.append(get_sp_cut_worker(x, p_s, h[scen_idx], T_s, scen_idx))

        sp_time = time.time() - sp_time

        # get dual values
        cut_time = time.time()

        Q = np.sum(list(map(lambda x: x[0], results)))
        a_sum = np.sum(list(map(lambda x: x[1], results)))
        b_sum = np.sum(list(map(lambda x: x[2], results)), axis=0)

        cut_time = time.time() - cut_time

        if theta >= Q - tol:
            print("Done: theta >= Q")
            break
    
        # otherwise add cut
        x_beta_sum_ = 0
        for i, x_var in master_problem._x.items():
            x_beta_sum_ += b_sum[i] * x_var

        master_problem.addConstr(master_problem._theta >= a_sum - x_beta_sum_, name=f"bc_{master_problem._bd_cuts}")
        master_problem._bd_cuts += 1
        
    time_benders = time.time() - time_benders
    master_problem._time_benders = time_benders
        
    return master_problem


# ## Functions for Alternating Cuts

def hash_x(x):
    """ Hashes solution to tuple.  Rounds values to avoid numerical issues.  """
    xh = x.values()
    for i in range(len(xh)):
        if xh[i] < 0.5:
            xh[i] = 0
        else:
            xh[i] = 1
    return tuple(xh)


def compute_subgradient_cut(master_problem, x, theta):
    """ Computes subgradient cuts based on IP solution of model. """
    
    # x to list of values
    x = list(map(lambda y: y, x.values()))
    x = list(map(lambda x: max(x, 0), x)) # fix value to be >= 0
    x = list(map(lambda x: min(x, 1), x)) # fix value to be <= 1
    
    # variables consistent between scenarios
    T_s = master_problem._T_s
    p_s = master_problem._p_s
    h = master_problem._h
    
    # solve all subproblems and recover duals.    
    sp_time = time.time()    
    results = []

    if master_problem._use_mp:
        for scen_idx in range(master_problem._n_scenarios):
            results.append(
                pool.apply_async(get_sp_cut_worker, (x, p_s, h[scen_idx], T_s, scen_idx)))
        results = [r.get() for r in results]

    else:
        for scen_idx in range(master_problem._n_scenarios):
            print(f"     Subgradient cut for scen {scen_idx}/{master_problem._n_scenarios}")
            results.append(get_sp_cut_worker(x, p_s, h[scen_idx], T_s, scen_idx))

    sp_time = time.time() - sp_time
    
    Q = np.sum(list(map(lambda x: x[0], results)))
    alpha = np.sum(list(map(lambda x: x[1], results)))
    beta = np.sum(list(map(lambda x: x[2], results)), axis=0)
    
    return Q, alpha, beta


def add_subgradient_cut(master_problem, alpha, beta):
    """ Adds subgradient cut to master_problem. """
    x_beta_sum_ = 0
    for i, x_var in master_problem._x.items():
        x_beta_sum_ += beta[i] * x_var
    master_problem.cbLazy(master_problem._theta >= alpha - x_beta_sum_)
    master_problem._sg_cuts += 1
    return master_problem


def get_optimality_cut_worker(x, p_s, scen_idx):
    """ Multiprocessing function for get sp cut. """
    
    # load subproblem, fix x, then optimize
    Q_s = master_problem._ip_subproblems[scen_idx]
    Q_s = fix_first_stage_sol(Q_s, x)
    Q_s.optimize()
    
    q_val = p_s * Q_s.objVal
    
    return q_val


def compute_integer_optimality_cut(master_problem, x, theta):
    """ Integer optimality cut. """
    # x to list of values
    x = list(map(lambda y: y, x.values()))
    x = list(map(lambda x: max(x, 0), x)) # fix value to be >= 0
    x = list(map(lambda x: min(x, 1), x)) # fix value to be <= 1
    
    # integer second-stage
    p_s = master_problem._p_s
    
    sp_time = time.time()
    results = []

    if master_problem._use_mp:
        for scen_idx in range(master_problem._n_scenarios):
            results.append(
                pool.apply_async(get_optimality_cut_worker, (x, p_s, scen_idx)))
        results = [r.get() for r in results]

    else:
        for scen_idx in range(master_problem._n_scenarios):
            print(f"     Optimality cut for scen {scen_idx}/{master_problem._n_scenarios}")
            results.append(get_optimality_cut_worker(x, p_s, scen_idx))

    sp_time = time.time() - sp_time
    
    Q = np.sum(results)

    return Q


def add_integer_optimality_cut(master_problem, x, Q):
    """ Add integer optimality cut to master_problem. """
    x = list(map(lambda y: y, x.values()))
    
    # compute the set S for integer cut
    S = []
    S_not = []
    for i, var in master_problem._x.items():
        if x[i] > 0.99:
            S.append(var)
        else:
            S_not.append(var)
            
    # compute and add integer cut
    integer_cut = (Q - master_problem._L) * (sum(S) - sum(S_not) - len(S)) + Q 
    master_problem.cbLazy(master_problem._theta >= integer_cut)
    master_problem._io_cuts += 1
    
    return master_problem


def optimality_cuts(master_problem, where):
    """ Callback function for optimality cuts.  """
    
    if where == gp.GRB.Callback.MIPNODE:
        n_nodes = master_problem.cbGet(gp.GRB.Callback.MIP_NODCNT)
        if n_nodes % 10 == 0:
            print("  # Nodes:", n_nodes)
        
    if where == gp.GRB.Callback.MIPSOL:
        
        x = master_problem.cbGetSolution(master_problem._x)
        theta = master_problem.cbGetSolution(master_problem._theta)
        
        print("Callback:")
    
        # condition for satisfied integer solution
        if hash_x(x) in master_problem._V:
            print("  Solution (x) in V, ending callback")
            return
        
        # if no sg cuts, then add them without adding integer cut
        if hash_x(x) not in master_problem._V_lp: 
            print("  Solution (x) not in V_lp...")
            
            # compute Q_lp, and subgradient cut info
            Q_lp, alpha, beta = compute_subgradient_cut(master_problem, x, theta)
            
            # add x to V_LP
            master_problem._V_lp.add(hash_x(x))
            
            # add sg cut and return if theta < Q_lp
            if theta < Q_lp:
                print("  Adding subgradient cut (theta < Q_lp), ending callback")
                master_problem = add_subgradient_cut(master_problem, alpha, beta)
                return
            
            print("  No subgradient cut needed (theta >= Q_lp)")
        
        # integer optimality cut Q value
        Q_ip = compute_integer_optimality_cut(master_problem, x, theta)
        
        # if sg cuts, then add integer cuts
        master_problem._V.add(hash_x(x))
        
        if theta < Q_ip:
            print("  Adding optimality (theta < Q_ip), ending callback")
            add_integer_optimality_cut(master_problem, x, Q_ip)
            return
        
        print("Did not meet any conditions!")




def get_args():
    """ Parser args for the script. """
    parser = argparse.ArgumentParser(description='Runs Integer-L Shaped with Alternating cuts.')

    # problem args
    parser.add_argument('--problem', type=str, default="cflp_10_10")
    parser.add_argument('--n_scenarios', type=int, default=100)
    parser.add_argument('--test_set', type=str, default="0",  help='Evaluate on a test set (unseen scenarios).')

    # data args
    parser.add_argument('--data_path', type=str, default="./data/",  help='Path to data.')

    # parallelism args (set to (n-1)-cores)
    parser.add_argument('--n_procs', type=int, default=4,  help='Number of procs for parallel subproblem solving.')
    parser.add_argument('--use_mp', type=int, default=1,  help='Indicator for using multiprocessing.  Used for debugging.')


    # optimization parameters
    parser.add_argument('--time_limit', type=int, default=3 * 3600, help='Time limit for solver.')
    parser.add_argument('--mipgap', type=float, default=1e-6, help='Gap limit for solver.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for MIP solver.')
    
    args = parser.parse_args()

    return args



if __name__ == "__main__":

    # get args
    args = get_args()

    # load all two_sp specific information
    get_path = factory_get_path(args.problem)
    fp_inst = get_path(args.data_path, args.problem, ptype="inst", suffix=".pkl")

    with open(fp_inst, 'rb') as p:
        inst = pkl.load(p)

    two_sp = factory_two_sp(args.problem, inst)
    scenarios = two_sp.get_scenarios(args.n_scenarios, args.test_set)

    # initialize master problem
    master_problem = get_master_lp(args, two_sp)
    master_problem._n_scenarios = args.n_scenarios
    master_problem._use_mp = args.use_mp

    # get linear subproblems
    lp_subproblems = get_subproblems(args, two_sp, scenarios, lp=True, add_ub=True)
    master_problem._lp_subproblems = lp_subproblems

    # get integer subproblems
    ip_subproblems = get_subproblems(args, two_sp, scenarios, lp=False, add_ub=True)
    master_problem._ip_subproblems = ip_subproblems

    # add second-stage info
    master_problem = add_second_stage_info_to_mp(master_problem)

    # Compute lower bound
    print("Computing Lower Bound...")

    time_lower_bound = time.time()
    L =  compute_lower_bound(args, master_problem, two_sp, lp=False)
    L -= 1e-3
    time_lower_bound = time.time() - time_lower_bound

    print("  Lower bound (Q_IP):", L)
    print("  Time for lower bound:", time_lower_bound)
    print('IP LB:', L)

    set_lb(master_problem, L)

    # info for integer-L shaped with alternating cuts
    master_problem._bd_cuts = 0
    master_problem._sg_cuts = 0
    master_problem._io_cuts = 0
    master_problem.Params.lazyConstraints = 1

    master_problem._V = set()
    master_problem._V_lp = set()

    # initialize mp pool for benders/alternating cuts
    if master_problem._use_mp:
        pool = mp.Pool(processes=args.n_procs)

    ## BENDERS
    master_problem = benders(master_problem)
    benders_obj = master_problem.objVal


    print("Objective of LP Relaxation:", benders_obj)
    print("Solving time for LP Relaxation:", master_problem._time_benders)


    ## ALTERNATING CUTS
    # fix variables to integer
    for _, var in master_problem._x.items():
        var.vtype = "B"


    # optimize with callback
    master_problem.optimize(optimality_cuts)

    print("Bender's cuts:", master_problem._bd_cuts)
    print("Subgradient cuts:", master_problem._sg_cuts)
    print("Integer cuts:", master_problem._io_cuts)

    print("Number of Nodes:", master_problem.NodeCount)

    print("Benders time:", master_problem._time_benders)
    print("Time ILS:", master_problem.RunTime)
    print("Time total:", master_problem._time_benders + master_problem.RunTime)


    sol = {}

    for k, v in master_problem._x.items():
        sol[f"x_{k}"] = v.x

    fs_obj = two_sp.evaluate_first_stage_sol(sol, args.n_scenarios, test_set=args.test_set, n_procs=args.n_procs)
    print("First-stage solution obj:", fs_obj)

    # collect and store all results
    results = {
        'time_lower_bound' : time_lower_bound,
        'time_benders' : master_problem._time_benders,
        'time_integer_l_shaped' : master_problem.RunTime,
        'time_total' : time_lower_bound + master_problem._time_benders + master_problem.RunTime,

        'obj_benders' : benders_obj,
        'obj_fs' : fs_obj,

        'cuts_benders' : master_problem._bd_cuts,
        'cuts_subgradient' : master_problem._sg_cuts,
        'cuts_integer_opt' : master_problem._io_cuts,
        
        'n_nodes' : master_problem.NodeCount,
    }

    problem_str = f"s{args.n_scenarios}_ts{args.test_set}"
    fp_results = get_path(args.data_path, args.problem, ptype=f"ils_{problem_str}", suffix=".pkl")

    with open(fp_results, 'wb') as p:
        pkl.dump(results, p)
