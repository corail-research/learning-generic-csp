from typing import List
import torch
from torch_geometric.data import HeteroData


def parse_dimacs_cnf(filepath:str):
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    problem_definition = lines[0].strip().split()
    num_literals = problem_definition[2]
    num_clauses = problem_definition[3]
    
    is_sat = filepath[-8]
    clauses = []
    for i in range(1, len(lines)):
        line = lines[i].strip().split()[:-1]
        new_clause = Clause(line)
        clauses.append(new_clause)
    
    return CNF(clauses, is_sat)

class Clause:
    """
    Simple class representing an OR clause; i.e. a line in a CNF problem.
    """
    def __init__(self, line:List[str]):
        self.variables = {int(var) for var in line}

class CNF:
    def __init__(self, clauses:List[Clause], is_sat:int):
        self.clauses = clauses
        self.is_sat = is_sat
        variables = set()
        for clause in self.clauses:
            base_vars = {abs(var) - 1 for var in clause.variables}
            variables = variables.union(base_vars)
        variables = list(variables)
        self.variables = sorted(variables)
    
    def build_heterogeneous_graph(self):
        """
        Build heterogeneous graph representation. The graph's label will be stored in data["variable"].y
        """
        # Nodes and node stuff
        constraints = [[1, 0]]
        negation_operator_id = 0
        operators = []

        # Edges and edge stuff
        variable_to_value_edges = self.get_sat_variable_to_domain_edges(self.variables)
        variable_to_operator_edges = []
        variable_to_constraint_edges = []
        operator_to_constraint_edges = []
        constraint_to_constraint_edges = []

        for i, clause in enumerate(self.clauses):
            current_constraint_index = i + 1
            constraints.append([0, 1])
            for variable in clause.variables:
                variable_index = abs(variable) - 1
                if variable < 0:
                    operators.append([1])
                    variable_to_operator_edges.append([variable_index, negation_operator_id])
                    variable_to_operator_edges.append([variable_index, negation_operator_id])
                    operator_to_constraint_edges.append([negation_operator_id, current_constraint_index])
                    negation_operator_id += 1
                else:
                    variable_to_constraint_edges.append([variable_index, current_constraint_index])
            constraint_to_constraint_edges.append([0, current_constraint_index])
        data = HeteroData()
        data["variable"].x = torch.Tensor([[1] for _ in self.variables])
        label = [0, 1] if self.is_sat else [1, 0]
        data["variable"].y = torch.Tensor([label])
        data["value"].x = torch.Tensor([[0], [1]])
        data["operator"].x = torch.Tensor([[1] for _ in range(negation_operator_id)])
        data["constraint"].x = torch.Tensor(constraints)

        data["variable", "connected_to", "value"].edge_index = self.build_edge_index_tensor(variable_to_value_edges)
        data["variable", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(variable_to_operator_edges)
        data["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(variable_to_constraint_edges)
        data["operator", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(operator_to_constraint_edges)
        data["constraint", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(constraint_to_constraint_edges)

        return data

    def get_sat_variable_to_domain_edges(self, variables):
        edges = []
        for variable in variables:
            for value in [0, 1]:
                new_edge = [variable, value]
                edges.append(new_edge)
        return edges
    
    def build_edge_index_tensor(self, edges:List)->torch.Tensor:
        return torch.Tensor(edges).long().t().contiguous()