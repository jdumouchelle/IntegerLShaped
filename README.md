# IntegerLShaped
A Python implementation of the integer L-shaped method for solving two-stage stochastic programs.  Implementation is based on the alternating cut variant of the integer L-shaped method from (Angulo et al., 2016).  


## Example
This is an example of running on SSLP_5_25_50 from the SIPLIB instances.  
```
python -m ils.scripts.generate_instance --problem sslp_5_25
python -m ils.scripts.ils_sslp --problem sslp_5_25 --n_scenarios 50 --test_set siplib --threads [N]
```



## References
- Laporte, G., & Louveaux, F. V. (1993). The integer L-shaped method for stochastic integer programs with complete recourse. ***Operations research letters***, 13(3), 133-142.
- Angulo, G., Ahmed, S., & Dey, S. S. (2016). Improving the integer L-shaped method. ***INFORMS Journal on Computing***, 28(3), 483-499.
