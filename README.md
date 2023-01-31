# Graph-Representation
This repo contains experiments related to the representation of combinatorial optimization problems as (heterogeneous) graphs. To start, the problems we will represent are the ones related to the mini-solver track of the [XCSP3 competition](https://www.xcsp.org/). In the mini-track, the problems are limited to the following constraints:

- intension
- extension (table)
- allDifferent
- sum
- element

As a start, we will represent the problems using a heterogeneous graph with 5 node types:
- Constraint. Constraint nodes will have, among other features, a 1-hot encoding that represents the type of constraint. Constraint nodes represent constraints on input variables, such as equality, greater than, not equal, etc.
- Operators. Operator nodes combine other variables and/or other operator nodes with an operation. Examples of operations include: sum, subtraction, multiplication, tuple (as in an extension constraint). 
- Variables. Variables represent the problem's variables
- Domain (values). Represent values that nodes can take.
- Constants. Numerical values that are not Domain values.
