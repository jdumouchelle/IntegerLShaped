import os
import pickle as pkl
import numpy as np
import gurobipy as gp

from .instance_generator import InstanceGenerator


class SSLPGenerator(InstanceGenerator):

    def __init__(self, n_locations, n_clients, data_path, siplib_instance_names):
        """ Constructor for SSLP instance generator.  """
        self.n_clients = n_clients
        self.n_locations = n_locations
        self.data_path = data_path
        self.siplib_instance_names = siplib_instance_names

    def generate_instance(self):
        """
        Generate a Stochastic Server Location Problem instance.
        Outputs as a gurobi model.
        """
        print("Generating instance...")

        inst = {}
        inst['n_locations'] = self.n_locations
        inst['n_clients'] = self.n_clients
        inst['siplib_instance_names'] = self.siplib_instance_names

        self._create_siplib_lp_files(inst, self.data_path)

        self._get_obj_data(inst)
        self._get_constr_data(inst)
        self._get_siplib_scenarios(inst)

        return inst

    #-----------------------------------------------------#
    #       SIPLIB Instance reading functions.            #
    #-----------------------------------------------------#

    @staticmethod
    def _create_siplib_lp_files(inst, data_path):
        """ Gets the EF file form the SMPS and MPS data. """
        lp_dir = data_path + '/sslp/sslp_data/'
        inst['siplib_instance_path_dict'] = {}

        for sslp_instance in inst['siplib_instance_names']:

            lp_smps_file = lp_dir + sslp_instance + '.smps'
            lp_mps_file = lp_dir + sslp_instance + '.mps'

            # create .mps file if it does not exist
            if not os.path.isfile(lp_smps_file):
                f_content = [f"{sslp_instance}.cor\n", f"{sslp_instance}.tim\n", f"{sslp_instance}.sto"]
                with open(lp_smps_file, 'w') as f:
                    f.writelines(f_content)

            # load smps file to scip
            if not os.path.isfile(lp_mps_file):
                import pyscipopt
                model = pyscipopt.Model()
                model.readProblem(lp_smps_file)
                model.writeProblem(lp_mps_file)

            inst['siplib_instance_path_dict'][sslp_instance] = lp_mps_file


    @staticmethod
    def _get_obj_data(inst):
        """ Gets the objective function information from the extensive form. """
        sslp_instance = inst['siplib_instance_names'][0]
        sslp_instance_fp = inst['siplib_instance_path_dict'][sslp_instance]
        n_scenarios_in_instance = int(sslp_instance.split("_")[-1])

        model = gp.read(sslp_instance_fp)

        # initialize data structure to store scenario costs
        first_stage_costs = np.zeros(inst['n_locations'])
        second_stage_costs = np.zeros((inst['n_clients'], inst['n_locations']))
        recourse_costs = np.zeros(inst['n_locations'])

        # recover costs for second stage variables
        for var in model.getVars():
            if "y" in var.varName:
                v = var.varName.split("_")
                client = int(v[1]) - 1
                location = int(v[2]) - 1
                scenario = int(v[4])
                second_stage_costs[client][location] = var.obj * n_scenarios_in_instance

            elif "x" in var.varName and var.varName.count("_") > 1:
                v = var.varName.split("_")
                location = int(v[1]) - 1
                scenario = int(v[4])
                recourse_costs[location] = var.obj * n_scenarios_in_instance

            else:
                v = var.varName.split("_")
                location = int(v[1]) - 1
                first_stage_costs[location] = var.obj

        inst['first_stage_costs'] = first_stage_costs
        inst['second_stage_costs'] = second_stage_costs
        inst['recourse_costs'] = recourse_costs


    @staticmethod
    def _get_constr_data(inst):
        """ Gets constraint coefficients. """
        sslp_instance = inst['siplib_instance_names'][0]
        sslp_instance_fp = inst['siplib_instance_path_dict'][sslp_instance]
        n_scenarios_in_instance = int(sslp_instance.split("_")[-1])

        model = gp.read(sslp_instance_fp)

        # create dicts for all variables
        x_vars, y_vars, r_vars = {}, {}, {}
        for var in model.getVars():
            if "y" in var.varName:
                y_vars[var.varname] = var
            elif "x" in var.varName and var.varName.count("_") > 1:
                r_vars[var.varname] = var
            else:
                x_vars[var.varname] = var

        # initialize data structures for constraint data
        loc_limit = 0
        client_coeffs = np.zeros((inst['n_clients'], inst['n_locations']))
        location_coeffs = np.zeros(inst['n_locations'])
        recourse_coeffs = np.zeros(inst['n_locations'])

        # collect cconstraint data
        for constr in model.getConstrs():
            if is_fs_constr(model, constr, x_vars, inst['n_locations']):
                loc_limit = - constr.RHS

            elif is_client_constr(constr):
                pass

            elif is_demand_constr(constr):

                scenario, location, location_coeff, client_coeff, recourse_coeff = get_demand_constr_data(
                    model, constr, x_vars, y_vars, r_vars, inst['n_locations'], inst['n_clients'])

                location_coeffs[location] = location_coeff
                client_coeffs[:, location] = client_coeff
                recourse_coeffs[location] = recourse_coeff

            else:
                raise Exception(f'Constraint {constr} not handled.')

        # store cconstraint data
        inst['location_limit'] = loc_limit
        inst['client_coeffs'] = client_coeffs
        inst['location_coeffs'] = location_coeffs
        inst['recourse_coeffs'] = recourse_coeffs


    @staticmethod
    def _get_siplib_scenarios(inst):

        inst['siplib_scenario_dict'] = {}

        for sslp_instance in inst['siplib_instance_names']:

            sslp_instance_fp = inst['siplib_instance_path_dict'][sslp_instance]
            n_scenarios = int(sslp_instance.split('_')[-1])
            clients_active = np.zeros((n_scenarios, inst['n_clients']))

            model = gp.read(sslp_instance_fp)

            # create dicts for all variables
            x_vars, y_vars, r_vars = {}, {}, {}
            for var in model.getVars():
                if "y" in var.varName:
                    y_vars[var.varname] = var
                elif "x" in var.varName and var.varName.count("_") > 1:
                    r_vars[var.varname] = var
                else:
                    x_vars[var.varname] = var

            for constr in model.getConstrs():
                if is_fs_constr(model, constr, x_vars, inst['n_locations']):
                    pass
                elif is_client_constr(constr):
                    scenario, client, rhs = get_client_constr_rhs(model, constr, y_vars, inst['n_locations'],
                                                                  inst['n_clients'])
                    clients_active[scenario][client] = rhs

            inst['siplib_scenario_dict'][sslp_instance] = clients_active




#--------------------------------------------------------------------------------------------------#
#               Some utilitiy functions for collecting data from the constraint matrix             #
#--------------------------------------------------------------------------------------------------#

def is_fs_constr(model, constr, x_vars, n_locations):
    for location_index in range(n_locations):
        if model.getCoeff(constr, x_vars[f'x_{location_index + 1}']) == 0:
            return False
    return True


def is_client_constr(constr):
    return constr.sense == '='


def get_client_constr_rhs(model, constr, y_vars, n_locations, n_clients):
    scenario = int(constr.constrName.split("_")[-1])

    client = -1
    for c in range(1, n_clients + 1):
        if model.getCoeff(constr, y_vars[f"y_{c}_1_1_{scenario}"]) == 1:
            rhs = constr.RHS
            client = c - 1
            break

    return scenario, client, rhs


def is_demand_constr(constr):
    return constr.sense == '>' and constr.rhs == 0


def get_demand_constr_data(model, constr, x_vars, y_vars, r_vars, n_locations, n_clients):
    # get scenaio
    scenario = int(constr.constrName.split("_")[-1])

    # get location
    location = -1
    for l in range(1, n_locations + 1):
        if model.getCoeff(constr, x_vars[f"x_{l}"]) != 0:
            location = l - 1
            break
    location_coeff = model.getCoeff(constr, x_vars[f"x_{location + 1}"])

    # get client coeffs
    client_coeffs = np.zeros(n_clients)
    for client in range(n_clients):
        v_name = f"y_{client + 1}_{location + 1}_1_{scenario}"
        client_coeffs[client] = model.getCoeff(constr, y_vars[v_name])

    v_name = f"x_{location + 1}_0_1_{scenario}"
    recourse_coeff = model.getCoeff(constr, r_vars[v_name])

    return scenario, location, location_coeff, client_coeffs, recourse_coeff
