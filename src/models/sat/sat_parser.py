from networkx import from_scipy_sparse_matrix, betweenness_centrality, eigenvector_centrality, closeness_centrality
from typing import List
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import numpy as np


def parse_dimacs_cnf(filepath:str):
    with open(filepath, "r") as f:
        lines = f.readlines()
    
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
    def __init__(self, clauses:List[Clause], is_sat:str):
        self.clauses = clauses
        self.is_sat = int(is_sat)
        variables = set()
        base_variables = set()
        for clause in self.clauses:
            base_vars = {abs(var) -1 for var in clause.variables}
            base_variables = base_variables.union(base_vars)
            variables = variables.union(clause.variables)
        variables = list(variables)
        base_variables = list(base_variables)
        self.variables = sorted(variables)
        self.base_variables = sorted(base_variables)
    
    def build_heterogeneous_graph(self, original=True):
        """Build the modified graph representation; i.e. one where variables (literals) are directly connected to their negated variable (literals)

        Returns:
            data (torch_geometric.data.HeteroData): graph for the SAT problem
        """
        # Nodes and node stuff
        constraints = [[1, 0]]
        negation_operator_id = 0
        operators = []

        # Edges and edge stuff
        variable_to_value_edges = self.get_sat_variable_to_domain_edges(self.base_variables)
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
                    operator_to_constraint_edges.append([negation_operator_id, current_constraint_index])
                    negation_operator_id += 1
                else:
                    variable_to_constraint_edges.append([variable_index, current_constraint_index])
            constraint_to_constraint_edges.append([0, current_constraint_index])
        data = HeteroData()

        var_tensor = torch.Tensor([[1, len(self.base_variables), len(self.clauses)] for _ in self.base_variables])

        data["variable"].x = var_tensor
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
        T.ToUndirected()(data)

        return data
    
    def build_sat_specific_heterogeneous_graph(self, random_values=False):
        """Build the modified graph representation; i.e. one where variables (literals) are directly connected to their negated variable (literals)
        IMPORTANT: After investigation, it was concluded that this formulation is not generic enough as it leverages SAT-specific structure. 

        Returns:
            data (torch_geometric.data.HeteroData): graph for the SAT problem
        """
        # Nodes and node stuff
        variable_to_index = {var: idx for idx, var in enumerate(self.variables)}
        constraints = [[1, 0]]
        negation_operator_id = 0
        operators = []

        # Edges and edge stuff
        variable_to_value_edges = self.get_sat_variable_to_domain_edges(self.variables, modified=True)
        variable_to_operator_edges = []
        variable_to_constraint_edges = []
        variable_to_operator_edges = []
        constraint_to_constraint_edges = []

        for i, clause in enumerate(self.clauses):
            current_constraint_index = i + 1
            constraints.append([0, 1])
            for variable in clause.variables:
                variable_index = variable_to_index[variable]
                if variable < 0:
                    positive_var = abs(variable)
                    positive_var_index = variable_to_index.get(positive_var, None)
                    if positive_var_index is not None:
                        pairs_to_add = [
                            [positive_var_index, negation_operator_id],
                            [variable_index, negation_operator_id]
                        ]
                        if pairs_to_add[0] not in variable_to_operator_edges and pairs_to_add[1] not in variable_to_operator_edges:
                            variable_to_operator_edges.append(pairs_to_add[0])
                            variable_to_operator_edges.append(pairs_to_add[1])                    
                            negation_operator_id += 1
                    
                variable_to_constraint_edges.append([variable_index, current_constraint_index])
            constraint_to_constraint_edges.append([0, current_constraint_index])
        data = HeteroData()
        if random_values:
            var_tensor = torch.randn((len(self.variables), 1))
        else:
            var_tensor = torch.Tensor([[1] if var > 0 else [-1] for var in self.variables])
        data["variable"].x = var_tensor
        label = [0, 1] if self.is_sat else [1, 0]
        data["variable"].y = torch.Tensor([label])
        data["value"].x = torch.Tensor([[0], [1]])
        data["operator"].x = torch.Tensor(
            [[1] for _ in range(negation_operator_id)])
        data["constraint"].x = torch.Tensor(constraints)

        data["variable", "connected_to", "value"].edge_index = self.build_edge_index_tensor(variable_to_value_edges)
        data["variable", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(variable_to_operator_edges)
        data["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(variable_to_constraint_edges)
        data["constraint", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(constraint_to_constraint_edges)
        T.ToUndirected()(data)

        return data


    def build_generic_heterogeneous_graph(self, use_node_id_as_variable_feature=False, spatial_dimension=2):
        """Build generic graph representation with graph refactoring. This is the same representation as build_heterogeneous_graph, but uses one
        negation operator per literal instead of creating one for every negation that appears.
        Args:
            use_node_id_as_feature (bool): if set to true, the variable node features will be their id (variable x_2 will have value 2)
        Returns:
            data (torch_geometric.data.HeteroData): graph for the SAT problem
        """
        # Nodes and node stuff
        constraints = []
        operators = [[-1] for _ in self.base_variables]
        meta = [[len(self.clauses), len(self.base_variables)]]

        # Edges and edge stuff
        variable_to_value_edges = self.get_sat_variable_to_domain_edges(self.base_variables)
        variable_to_operator_edges = []
        variable_to_constraint_edges = []
        operator_to_constraint_edges = []
        meta_to_constraint_edges = []

        for i, clause in enumerate(self.clauses):
            current_constraint_index = i
            constraints.append([1, len(clause.variables), i])
            for variable in clause.variables:
                variable_index = abs(variable) - 1
                if variable < 0:
                    if [variable_index, variable_index] not in variable_to_operator_edges:
                        variable_to_operator_edges.append([variable_index, variable_index]) # the operator index is the same as the variable index
                    operator_to_constraint_edges.append([variable_index, current_constraint_index])
                else:
                    variable_to_constraint_edges.append([variable_index, current_constraint_index])
            meta_to_constraint_edges.append([0, current_constraint_index])
        
        data = HeteroData()
        if use_node_id_as_variable_feature:
            var_tensor = torch.Tensor([[i] for i in self.base_variables])
        else:
            var_tensor = torch.Tensor([[1] for _ in self.base_variables])

        data["variable"].x = var_tensor
        label = [0, 1] if self.is_sat else [1, 0]
        data["variable"].y = torch.Tensor([label])
        data["value"].x = torch.Tensor([[0], [1]])
        data["operator"].x = torch.Tensor(operators)
        data["constraint"].x = torch.Tensor(constraints)
        data["meta"].x = torch.Tensor(meta)

        data["variable", "connected_to", "value"].edge_index = self.build_edge_index_tensor(variable_to_value_edges)
        data["variable", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(variable_to_operator_edges)
        data["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(variable_to_constraint_edges)
        data["operator", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(operator_to_constraint_edges)
        data["meta", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(meta_to_constraint_edges)
        
        centrality_measures = self.calculate_centrality_measures(data) # Calculate centrality measures for all nodes
        spatial_encodings = self.generate_spatial_encoding(num_nodes=len(centrality_measures), dimension=spatial_dimension) # Generate spatial encodings for all node types

        # Update the features for each node type
        var_start_idx = 0
        var_end_idx = len(self.base_variables)
        data["variable"].x = self.update_node_features("variable", var_tensor, centrality_measures, spatial_encodings, var_start_idx, var_end_idx)

        value_start_idx = var_end_idx
        value_end_idx = value_start_idx + 2
        data["value"].x = self.update_node_features("value", data["value"].x, centrality_measures, spatial_encodings, value_start_idx, value_end_idx)

        operator_start_idx = value_end_idx
        operator_end_idx = operator_start_idx + len(operators)
        data["operator"].x = self.update_node_features("operator", data["operator"].x, centrality_measures, spatial_encodings, operator_start_idx, operator_end_idx)

        constraint_start_idx = operator_end_idx
        constraint_end_idx = constraint_start_idx + len(constraints)
        data["constraint"].x = self.update_node_features("constraint", data["constraint"].x, centrality_measures, spatial_encodings, constraint_start_idx, constraint_end_idx)

        meta_start_idx = constraint_end_idx
        meta_end_idx = meta_start_idx + 1
        data["meta"].x = self.update_node_features("meta", data["meta"].x, centrality_measures, spatial_encodings, meta_start_idx, meta_end_idx)
        T.ToUndirected()(data)

        return data
    
    def calculate_centrality_measures(self, data):
        adj_matrix = pyg_utils.to_networkx(data).to_undirected().to_adjacency_matrix().tocoo()
        nx_graph = from_scipy_sparse_matrix(adj_matrix)
        betw_cent = betweenness_centrality(nx_graph)
        eigv_cent = eigenvector_centrality(nx_graph)
        close_cent = closeness_centrality(nx_graph)

        # Normalize centrality measures
        betw_cent = np.array([betw_cent[i] for i in sorted(betw_cent.keys())])
        eigv_cent = np.array([eigv_cent[i] for i in sorted(eigv_cent.keys())])
        close_cent = np.array([close_cent[i] for i in sorted(close_cent.keys())])

        return np.vstack((betw_cent, eigv_cent, close_cent)).T

    def generate_spatial_encoding(self, num_nodes, dimension):
        encoding = np.random.uniform(size=(num_nodes, dimension))
        return encoding

    def get_sat_variable_to_domain_edges(self, variables, modified=False):
        edges = []
        for i, variable in enumerate(variables):
            for value in [0, 1]:
                if modified:
                    new_edge = [i, value]
                else:
                    new_edge = [variable, value]
                edges.append(new_edge)
        return edges
    
    def build_edge_index_tensor(self, edges:List)->torch.Tensor:
        return torch.Tensor(edges).long().t().contiguous()