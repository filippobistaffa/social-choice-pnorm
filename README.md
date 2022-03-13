A Norm Approximation Approach to Account for Multiple Ethical Principles in Social Choice
===================
This repository contains the implementation of all the algorithms and data of the experimental section of
"A Norm Approximation Approach to Account for Multiple Ethical Principles in Social Choice"
by Francisco Salas-Molina, Filippo Bistaffa, and Juan A. Rodríguez-Aguilar.

Dependencies
----------
 - [Python 3](https://www.python.org/downloads/)
 - [CVXPY](https://www.cvxpy.org/)
 - [IBM CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)

Dataset
----------
All experiments consider the dataset discussed by Nordström *et al.* in "Aggregation of Preferences in Participatory Forest Planning with Multiple Citeria: an Application to the Urban Forest in Lycksele, Sweden"
([article](https://doi.org/10.1139/X09-107)).

Execution
----------
Our single-norm approach must be executed by means of the [`single_norm.py`](single_norm.py) Python script, i.e.,
```
usage: single_norm.py [-h] [-n N] [-m M] [-p P] [-e E] [-w W] [-b B] [-i I] [-o O] [-u] [-l] [-t]

optional arguments:
  -h, --help  show this help message and exit
  -n N        number of stakeholders (default: 7)
  -m M        size of preference matrix (default: 5)
  -p P        p-norm
  -e E        epsilon used to compute limit p
  -w W        CSV file with weights
  -b B        CSV file with b vector
  -i I        computes equivalent p given an input consensus
  -o O        write consensus to file
  -u          optimize only upper-triangular
  -l          compute the limit p
  -t          compute the threshold p
```
Our multi-norm approach must be executed by means of the [`multi_norm.py`](multi_norm.py) Python script, i.e.,
```
usage: multi_norm.py [-h] [-n N] [-m M] [-w W] [-b B] [-p P [P ...]]
                     [-l L [L ...]] [-o O] [-u] [-v] [-P] [-M] [--no-weights]

optional arguments:
  -h, --help    show this help message and exit
  -n N          n
  -m M          m
  -w W          CSV file with weights
  -b B          CSV file with b vector
  -p P [P ...]  p-norms
  -l L [L ...]  lambdas
  -o O          write consensus to file
  -u            optimize only upper-triangular
  -v            verbose mode
  -P            print LaTeX code for PGFPLOTS boxplot
  -M            perform the Mann-Whitney U test
  --no-weights  do not weight norms
```
