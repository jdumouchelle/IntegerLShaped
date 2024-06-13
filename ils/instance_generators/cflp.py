import time
import numpy as np
import pickle as pkl

from .instance_generator import InstanceGenerator


class CFLPGenerator(InstanceGenerator):

    def __init__(self, n_facilities, n_customers, seed):
        """ Constructor for CFLP instance generator. """
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        self.seed_used = seed

        self.rng = np.random.RandomState()
        self.rng.seed(seed)


    def generate_instance(self):
        """
        Generate a Capacited Facility Location problem following
            Cornuejols G, Sridharan R, Thizy J-M (1991)
            A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
            European Journal of Operations Research 50:280-297.
        Outputs as a gurobi model.
        """
        print("Generating instance...")

        inst = {}
        inst['n_facilities'] = self.n_facilities
        inst['n_customers'] = self.n_customers

        # Fixed parameters.  Can be changed if desired.
        inst['integer_second_stage'] = True
        inst['bound_tightening_constrs'] = True
        inst['ratio'] = 2.0

        self._generate_first_stage_data(inst, self.rng)

        return inst


    def _generate_first_stage_data(self, inst, rng):
        """ Computes and stores information for first stage problem. """
        inst['c_x'] = rng.rand(self.n_customers)
        inst['c_y'] = rng.rand(self.n_customers)

        inst['f_x'] = rng.rand(self.n_facilities)
        inst['f_y'] = rng.rand(self.n_facilities)

        inst['demands'] = rng.randint(5, 35 + 1, size=self.n_customers)
        inst['capacities'] = rng.randint(10, 160 + 1, size=self.n_facilities)
        inst['fixed_costs'] = rng.randint(100, 110 + 1, size=self.n_facilities) * np.sqrt(inst['capacities']) \
                              + rng.randint(90 + 1, size=self.n_facilities)
        inst['fixed_costs'] = inst['fixed_costs'].astype(int)

        inst['total_demand'] = inst['demands'].sum()
        inst['total_capacity'] = inst['capacities'].sum()

        # adjust capacities according to ratio
        inst['capacities'] = inst['capacities'] * inst['ratio'] * inst['total_demand'] / inst['total_capacity']
        inst['capacities'] = inst['capacities'].astype(int)
        inst['total_capacity'] = inst['capacities'].sum()

        # transportation costs
        inst['trans_costs'] = np.sqrt(
            (inst['c_x'].reshape((-1, 1)) - inst['f_x'].reshape((1, -1))) ** 2
            + (inst['c_y'].reshape((-1, 1)) - inst['f_y'].reshape((1, -1))) ** 2) \
                              * 10 * inst['demands'].reshape((-1, 1))
        inst['trans_costs'] = inst['trans_costs'].transpose()

