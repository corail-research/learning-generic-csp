# Graph-Representation
This repo contains experiments related to the representation of combinatorial optimization problems as (heterogeneous) graphs. It contains our implementation of (NeuroSAT)[https://arxiv.org/abs/1802.03685] -a GNN trained to predict whether a given SAT problem is solvable or not- and an attempt to model SAT problems with a generic graph representation. The generic graph representation is intended to be used to model any combinatorial optimization as a graph.

## Our heterogeneous graph representation
As a start, we will represent the problems using a heterogeneous graph with 5 node types:
- Constraint. Constraint nodes will have, among other features, a 1-hot encoding that represents the type of constraint. Constraint nodes represent constraints on input variables, such as equality, greater than, not equal, etc.
- Operators. Operator nodes combine other variables and/or other operator nodes with an operation. Examples of operations include: sum, subtraction, multiplication, tuple (as in an extension constraint). 
- Variables. Variables represent the problem's variables
- Domain (values). Represent values that nodes can take.
- Constants. Numerical values that are not Domain values.

## Structure
For now, the project is still very small. It currently has the following folders:
- examples: contains jupyter notebooks with code examples for some problems
- sample_problems: contains some problem instances, as expressed with the XCSP3 format
- src: source code
- 

## Setup
As usual, start by creating a virtual environment and installing the requirements:
```
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
```
Then, check which GPUs are available and adjust your runs with this command:
```
CUDA_VISIBLE_DEVICES=[device-you-want-to-use] python3 -u [your-training-script]
```

find . -type d | grep -v './sample_problems\|./plots\|./sat_data\|./src/decision_tsp\|./venv\|./wandb\|./src/sat/generic_sat/data\|./src/sat/neurosat/wandb\'
                         
