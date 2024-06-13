# An Implementation of the Integer L-Shaped Method
A Python implementation of the integer L-shaped method for solving two-stage stochastic programs.  Implementation is based on the algorithm outlined in [Angulo 2016], and all of the implementation is done in Python with Gurobi as the MILP solver.  

This repository aims to provide a parallel implementation of the integer L-shaped method by solving the subproblems in parallel. The code is designed using Python multiprocessing, i.e., process-based parallelism, which is ideal computation on a single machine with multiple threads.  In addition, instance generators for the stochastic server location problem (based on the instances from [Ntaimo, 2005]) and stochastic capacitated facility location problem (based on the instances from [Dumouchelle, 2022]) are provided.  


## Example
The commands can be executed to run the integer L-shaped method on all SSLP instances.  
```
python -m ils.scripts.run_ils --problem sslp_5_25 --n_scenarios 50 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_5_25 --n_scenarios 100 --n_procs [N]

python -m ils.scripts.run_ils --problem sslp_10_50 --n_scenarios 50 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_10_50 --n_scenarios 100 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_10_50 --n_scenarios 500 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_10_50 --n_scenarios 1000 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_10_50 --n_scenarios 2000 --n_procs [N]

python -m ils.scripts.run_ils --problem sslp_15_45 --n_scenarios 5 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_15_45 --n_scenarios 10 --n_procs [N]
python -m ils.scripts.run_ils --problem sslp_15_45 --n_scenarios 15 --n_procs [N]
```



## Contributing & Adding New Problems

This repository was designed to support extensions to other problems.  Specifically, adding new two-stage stochastic programming problems should be relatively straightforward.  For a new problem `p`, the following changes must be made.
- `ils/utils/p.py`: This file includes functions to get paths to store instances/results.
- `ils/two_sp/p.py`: Implements a class that formulates subproblems, solves second-stage problems, and evaluates first-stage solutions.
- `ils/instance_generators/p.py`: Implements a class to generate instances.  Additional parameters/code may also need to be added to `ils/scripts/generate_instances.py`.
- `ils/ils/p.py`: Implements a class with useful functions for the integer L-shaped method.  Specifically, getting coefficients/dual values for subproblems will need to be implemented here.  In addition, functions for relaxation should be implemented.  To facilitate the best performance with Gurobi, redundant constraints must be added, and all dual values must be accessed in a single call to Gurobi.  Please see the implementations of SSLP/CFLP for examples.  

For all of the above, SSLP/CFLP should be useful examples to base implementation off when adding new problems.   


## Instances
If any instances are useful, please see the details 

- Stochastic Server Location Problem (SSLP):
  - Reference: [Ntaimo, 2005]
  - Link to original instances: https://www2.isye.gatech.edu/~sahmed/siplib/sslp/sslp.html
  - Notes: Instances can be downloaded via the above link and placed into the data directory.  Pickle files of the instances in a format ready to use for the integer L-shaped method are provided in `data` folder.

- Capacitated Facility Location Problem (CFLP):
  - Reference: [Dumouchelle, 2022]
  - Link to original instances: https://github.com/khalil-research/Neur2SP
  - Notes: These instances are generated similarly to the CFLP instances in [Dumouchelle, 2022].  The stochastic instances are generated based on the deterministic instances from [Cornuéjols, 1991].  Pickle files of the instances in a format ready to use for the integer L-shaped method are provided in `data` folder.  


## References
- Laporte, G., & Louveaux, F. V. (1993). The integer L-shaped method for stochastic integer programs with complete recourse. *Operations research letters*, 13(3), 133-142.
- Angulo, G., Ahmed, S., & Dey, S. S. (2016). Improving the integer L-shaped method. *INFORMS Journal on Computing*, 28(3), 483-499.
- Ntaimo, L., & Sen, S. (2005). The million-variable “march” for stochastic combinatorial optimization. Journal of Global Optimization, 32, 385-400.
- Dumouchelle, J., Patel, R. M., Khalil, E., & Bodur, M. (2022). Neur2SP: Neural two-stage stochastic programming. Advances in Neural Information Processing Systems, 35.
- Cornuéjols, G., Sridharan, R., & Thizy, J. M. (1991). A comparison of heuristics and relaxations for the capacitated plant location problem. European journal of operational research, 50(3), 280-297.


For any questions, please contact justin.dumouchelle@mail.utoronto.ca
