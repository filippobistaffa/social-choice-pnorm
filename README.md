
A norm approximation approach to account for multiple ethical principles in social choice
===================
This repository contains the implementation of all the algorithms and data of the experimental section of
"A Norm Approximation Approach to Account for Multiple Ethical Principles in Social Choice"
by Francisco Salas-Molina, Filippo Bistaffa, and Juan A. Rodríguez-Aguilar.

Dependencies
----------
 - [Python 3](https://www.python.org/downloads/)
 - [Julia](https://julialang.org/downloads/) and [PyJulia](https://pyjulia.readthedocs.io/en/latest/installation.html) (compiling a [custom system image](https://pyjulia.readthedocs.io/en/latest/sysimage.html) is required for faster initialisation)
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
usage: multi_norm.py [-h] [-n N] [-m M] [-w W] [-b B] [-p P] [-l L] [-o O] [-u] [-v]

optional arguments:
  -h, --help  show this help message and exit
  -n N        number of stakeholders (default: 7)
  -m M        size of preference matrix (default: 5)
  -w W        CSV file with weights
  -b B        CSV file with b vector
  -p P        CSV file with norms
  -l L        CSV file with lambdas
  -o O        write consensus to file
  -u          optimize only upper-triangular
  -v          verbose mode
```


Acknowledgements
----------
This repository contains the [implementation of the pIRLS algorithm](https://github.com/fast-algos/pIRLS) ([article](https://papers.nips.cc/paper/2019/hash/46c7cb50b373877fb2f8d5c4517bb969-Abstract.html)) and the [implementation of the General Norm Minimization Solver](https://github.com/yasumat/NormMinimization) ([article](http://www-infobiz.ist.osaka-u.ac.jp/wp-content/uploads/paper/pdf/e-heritage_ACCV2016_FGNA.pdf)). Both articles should be cited when citing our work.
