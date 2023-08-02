A General Approach for Computing a Consensus in Group Decision Making That Integrates Multiple Ethical Principles
===================
This repository contains the implementation of the algorithm presented by Francisco Salas-Molina, Filippo Bistaffa, and Juan A. Rodríguez-Aguilar in "[A General Approach for Computing a Consensus in Group Decision Making That Integrates Multiple Ethical Principles](https://www.sciencedirect.com/science/article/pii/S0038012123002069/pdfft?md5=bf631ed1e2094c91969e057c0f9b65b7&pid=1-s2.0-S0038012123002069-main.pdf)", Socio-Economic Planning Sciences volume 89, 2023, DOI: [10.1016/j.seps.2023.101694](https://doi.org/10.1016/j.seps.2023.101694).

Dependencies
----------
 - [Python 3.7](https://www.python.org/downloads/)
 - [CVXPY 1.2](https://www.cvxpy.org/)
 - [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) or [GUROBI](https://www.gurobi.com)

Dataset
----------
All experiments consider the dataset discussed by Nordström *et al.* in "Aggregation of Preferences in Participatory Forest Planning with Multiple Citeria: an Application to the Urban Forest in Lycksele, Sweden"
([article](https://doi.org/10.1139/X09-107)).

Execution
----------
Our multi-norm approach must be executed by means of the [`multi_norm.py`](multi_norm.py) Python script, i.e.,
```
usage: multi_norm.py [-h] [-n N] [-m M] [-w W] [-b B] [-p P [P ...]]
                     [-l L [L ...]] [-u] [-v] [-P] [-M] [-L] [-W]
                     [-S {CPLEX,GUROBI}]

options:
  -h, --help         show this help message and exit
  -n N               n
  -m M               m
  -w W               CSV file with weights
  -b B               CSV file with b vector
  -p P [P ...]       p-norms
  -l L [L ...]       lambdas
  -u                 optimize only upper-triangular
  -v                 verbose mode
  -P                 print LaTeX code for PGFPLOTS boxplot
  -M                 perform the Mann-Whitney U test
  -L                 print LaTeX code for stats
  -W                 do not weight norms
  -S {CPLEX,GUROBI}  choose solver (either CPLEX or GUROBI)
```
