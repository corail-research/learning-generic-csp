# Graph-Representation
This repo contains experiments related to the representation of combinatorial optimization problems as (heterogeneous) graphs. It contains our implementation of (NeuroSAT)[https://arxiv.org/abs/1802.03685] -a GNN trained to predict whether a given SAT problem is solvable or not- and an attempt to model SAT problems with a generic graph representation. The generic graph representation is intended to be used to model any combinatorial optimization as a graph.

## Our heterogeneous graph representation
As a start, we will represent the problems using a heterogeneous graph with 5 node types:
- Constraint. Constraint nodes will have, among other features, a 1-hot encoding that represents the type of constraint. Constraint nodes represent constraints on input variables, such as equality, greater than, not equal, etc.
- Operators. Operator nodes combine other variables and/or other operator nodes with an operation. Examples of operations include: sum, subtraction, multiplication, tuple (as in an extension constraint). 
Expand All
	@@ -20,7 +14,7 @@ For now, the project is still very small. It currently has the following folders
- examples: contains jupyter notebooks with code examples for some problems
- sample_problems: contains some problem instances, as expressed with the XCSP3 format
- src: source code

## SAT problem
The implementation for GNNs on the SAT problem can be found in src/models/sat. To complete the setup, you can run the scripts/setup.sh script to install PyMiniSolvers.

## Setup
As usual, start by creating a virtual environment and installing the requirements.

There are some aadditional installs you will need for this to be completely turn-key:

- Microsoft C++ Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/)


Then, check which GPUs are available and adjust your runs with this command:
```
CUDA_VISIBLE_DEVICES=[device-you-want-to-use] python3 -u [your-training-script]
```
