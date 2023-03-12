# Setup 
## QSopt
Concorde requires a linear programming solver. In this setup, we use QSopt. Simply download it (all three files) from (this link)[https://www.math.uwaterloo.ca/~bico/qsopt/downloads/downloads.htm]. I am using ubuntu, so I download all 3 files located at the bottom of the page and I place them in a directory named `qsopt_solver`. 

## Concorde Download, Installation and Setup
```
sudo apt update
sudo apt install build-essential libgmp-dev libgsl-dev
wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar -xvf co031219.tgz
cd concorde
./configure --prefix=/usr/local --with-qsopt=[full_path_to_qsopt_solver_directory]
make
export PATH="/usr/local/concorde/TSP:$PATH
```
Adjust the last line according to where you downloaded the concorde solver. Following this step by step will allow you to use concorde. This can be tested by running the following command:
```
concorde -s 99 -k 100
```
Running this command should yield the following:

```
concorde -s 99 -k 100
Host: [host_name]  Current process id: xxxx
Using random seed 99
Random 100 point set
XSet initial upperbound to 780 (from tour)
  LP Value  1: 738.500000  (0.00 seconds)
  LP Value  2: 765.000000  (0.02 seconds)
  LP Value  3: 774.660000  (0.05 seconds)
  LP Value  4: 778.000000  (0.09 seconds)
  LP Value  5: 778.465517  (0.13 seconds)
  LP Value  6: 778.705882  (0.16 seconds)
  LP Value  7: 779.538462  (0.20 seconds)
  LP Value  8: 779.937500  (0.24 seconds)
  LP Value  9: 780.000000  (0.26 seconds)
New lower bound: 780.000000
Final lower bound 780.000000, upper bound 780.000000
Exact lower bound: 780.000000
DIFF: 0.000000
Final LP has 180 rows, 336 columns, 2921 nonzeros
Optimal Solution: 780.00
Number of bbnodes: 1
Total Running Time: 0.45 (seconds)
```

## Installing pyconcorde
pyconcorde is a python wrapper for the concorde TSP solver. It can be installed by running the following commands:
```
git clone https://github.com/jvkersch/pyconcorde
cd pyconcorde
pip install -e .
```