# Graph-Representation
This repo contains experiments related to the representation of combinatorial optimization problems as (heterogeneous) graphs. To start, the problems we will represent are the ones related to the mini-solver track of the [XCSP3 competition](https://www.xcsp.org/). In the mini-track, the problems are limited to the following constraints:

- intension
- extension (table)
- allDifferent
- sum
- element

## Types of problems represented
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
- test: test code

## Setup
As usual, start by creating a virtual environment and installing the requirements:
```
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
```