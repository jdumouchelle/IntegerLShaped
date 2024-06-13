import pickle as pkl
from argparse import ArgumentParser

from ils.instance_generators import factory_instance_generator
from ils.utils import factory_get_path



#---------------------------------------------------------#
#                                                         #  
#     File to generate instance and save as .pkl files    #
#                                                         #   
#---------------------------------------------------------#



#-------------------------------------------#
#              Helper functions             #
#-------------------------------------------#

def get_generator_params(args):
    """ Gets params based on specified problem. """
    if 'cflp' in args.problem:
        _, n_facilities, n_customers = args.problem.split("_")
        params = {
            'n_facilities' : int(n_facilities),
            'n_customers' : int(n_customers),
            'seed' : args.seed,
        }

    elif 'sslp' in args.problem:
        _, n_locations, n_clients = args.problem.split("_")
        n_locations = int(n_locations)
        n_clients = int(n_clients)

        # get SSLP instances names
        if n_locations == 5 and n_clients == 25:
            siplib_instance_names = ['sslp_5_25_50', 'sslp_5_25_100']
        elif n_locations == 10 and n_clients == 50:
            siplib_instance_names = ['sslp_10_50_50', 'sslp_10_50_100', 'sslp_10_50_500', 
                                     'sslp_10_50_1000', 'sslp_10_50_2000']
        elif n_locations == 15 and n_clients == 45:
            siplib_instance_names = ['sslp_15_45_5', 'sslp_15_45_10', 'sslp_15_45_15']
        else:
            raise Exception("Invalid number of locations/clients for SSLP")

        params = {
            'n_locations' : n_locations,
            'n_clients' : n_clients,
            'data_path' : args.data_path,
            'siplib_instance_names' : siplib_instance_names,
        }

    return params



#-------------------------------------------#
#                   Main                    #
#-------------------------------------------#

def main(args):
    print(args.problem)
    get_path = factory_get_path(args.problem)

    params = get_generator_params(args)
    instance_generator = factory_instance_generator(args.problem, params)
    inst = instance_generator.generate_instance()

    fp_inst = get_path(args.data_path, args.problem, ptype="inst", suffix=".pkl")
    with open(fp_inst, 'wb') as p:
        pkl.dump(inst, p)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    main(args)
