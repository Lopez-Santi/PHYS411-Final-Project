Project Topics:
Economic Dynamics and Models, Coupled Oscillators, Network/Graph Theory, Stability and Criticallity

Project Description:
My goal in this project will be to explore the behavior of multiple economic systems under a specific
configuration. I will place a “Master economy” at the center of a scale free network to act as a sort
of external drive to the rest of the economies who are connected to it (and each other). The characteristics
of each economy will be dictated by using the Kaldor Business Cycle (maybe some type of oscillator
(van Der Pol maybe)) model and additional parameters will be utilized to make the model more realistic.
One specific behavior that I hope to identify is a critical locking radius, where the
economies located at a certain distance from the Master economy will stop locking. This distance of
the radius will depend on the parameters of the Master economy as well as the satellite economies
around it and I hope to investigate what this dependance is. Some numerical work will be utilized to
be able to model the system and look for patterns. Then an analytical approach will be used to simplify
the model and try to provide a “back of the envelope” calculation of the behavior.


#####
sim_helpers.py - contains functions to help with simulating the network and ODE solvers
plotting_helpers.py - contains functions to help with plotting the dynamics

Simple_econ_network.ipynb - contains an initial run of 3 different (2-D) models for a coupled network of econmies
Chaotic_econ_network.ipynb - contains an implementation of the chaotic business cycles described in 10.3.4
order_param_calc.ipynb - contains an implementation of how to calculate the order parameter (normalized amplitude of oscillations) as a function of the coupling strength

#####
Resources:
Main Resource - Introduction to Modern Dynamics (Oxford University Press, D. D. Nolte, 2019) Second Edition
ODE Solver Guide - https://scipy-cookbook.readthedocs.io/items/idx_ordinary_differential_equations.html
Chaos Models in Economic - https://arxiv.org/pdf/1001.3492.pdf
Dynamics of Coupled Van der Pol Oscillators - https://link.springer.com/article/10.1007/s10958-022-05860-6
Cascading failures in coupled networks: The critical role of node-coupling strength across networks - https://www.nature.com/articles/srep35352
Solving the Van Der Pol nonlinear diﬀerential equation using ﬁrst order approximation perturbation method - https://www.12000.org/my_notes/vanderpol/vanderpol_perturbation/index.htm
Van der Pol oscillator - http://www.scholarpedia.org/article/Van_der_Pol_oscillator

