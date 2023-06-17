from networkx import betweenness_centrality, eigenvector_centrality, closeness_centrality
from networkx import from_scipy_sparse_array
import networkx as nx
from typing import List
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt


def parse_dimacs_cnf(filepath:str):
    with open(filepath, "r") as f:
        lines = f.readlines()
    filename = filepath[filepath.find("sr_n"):]
    is_sat = filepath[-8]
    clauses = []
    for i in range(1, len(lines)):
        line = lines[i].strip().split()[:-1]
        new_clause = Clause(line)
        clauses.append(new_clause)
    
    return CNF(clauses, is_sat, filename)

class Clause:
    """
    Simple class representing an OR clause; i.e. a line in a CNF problem.
    """
    def __init__(self, line:List[str]):
        self.variables = {int(var) for var in line}

class CNF:
    def __init__(self, clauses:List[Clause], is_sat:str, filename:str=None):
        self.clauses = clauses
        self.is_sat = int(is_sat)
        self.filename = filename
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
    
    def build_sat_specific_heterogeneous_graph(self, use_sat_label_as_feature=False):
        """Build the modified graph representation; i.e. one where variables (literals) are directly connected to their negated variable (literals)
        IMPORTANT: After investigation, it was concluded that this formulation is not generic enough as it leverages SAT-specific structure. 
        Args:
            use_sat_label_as_feature (bool): Whether to use the label (sat or unsat) as a feature in the graph. This should be used for testing purposes only
        Returns:
            data (torch_geometric.data.HeteroData): graph for the SAT problem
        """
        # Nodes and node stuff
        variable_to_index = {var: idx for idx, var in enumerate(self.variables)}
        constraints = [[1, 0]]

        # Edges and edge stuff
        variable_to_variable_edges = []
        variable_to_constraint_edges = []
        constraint_to_constraint_edges = []

        for i, clause in enumerate(self.clauses):
            constraint_index = i + 1
            constraints.append([0, 1])
            for variable in clause.variables:
                variable_index = variable_to_index[variable]
                if variable < 0:
                    positive_var = abs(variable)
                    positive_var_index = variable_to_index.get(positive_var, None)
                    if positive_var_index is not None:
                        pair_to_add = [variable_index, positive_var_index]
                        if pair_to_add not in variable_to_variable_edges and pair_to_add[::-1] not in variable_to_variable_edges:
                            variable_to_variable_edges.append(pair_to_add)
                    
                variable_to_constraint_edges.append([variable_index, constraint_index])
            constraint_to_constraint_edges.append([0, constraint_index])
        
        data = HeteroData()
        if use_sat_label_as_feature:
            var_tensor = torch.Tensor([[1, self.is_sat] if var > 0 else [-1, self.is_sat] for var in self.variables])
        else:
            var_tensor = torch.Tensor([[1] if var > 0 else [-1] for var in self.variables])

        data["variable"].x = var_tensor
        label = [0, 1] if self.is_sat else [1, 0]
        data["variable"].y = torch.Tensor([label])
        data["constraint"].x = torch.Tensor(constraints)

        data["variable", "is_negation_of", "variable"].edge_index = self.build_edge_index_tensor(variable_to_variable_edges)
        data["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(variable_to_constraint_edges)
        data["constraint", "is_related_to", "constraint"].edge_index = self.build_edge_index_tensor(constraint_to_constraint_edges)
        data.filename = self.filename
        T.ToUndirected()(data)

        return data

    def build_generic_heterogeneous_graph(self, meta_connected_to_all=False):
        """Build generic graph representation with graph refactoring. This is the same representation as build_heterogeneous_graph, but uses one
        negation operator per literal instead of creating one for every negation that appears.
        Args:
            meta_connected_to_all (bool): if set to true, all nodes will be connected to the meta node. Otherwise, only the constraint nodes will
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
            constraints.append([1, len(clause.variables)/len(self.base_variables)])
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
        var_tensor = torch.Tensor([[1] for _ in self.base_variables])

        data["variable"].x = var_tensor
        label = 1 if self.is_sat else 0
        data["variable"].y = torch.tensor([label])
        data["value"].x = torch.Tensor([[0], [1]])
        data["operator"].x = torch.Tensor(operators)
        data["constraint"].x = torch.Tensor(constraints)
        data["meta"].x = torch.Tensor(meta)

        data["variable", "has_domain", "value"].edge_index = self.build_edge_index_tensor(variable_to_value_edges)
        data["variable", "affected_by", "operator"].edge_index = self.build_edge_index_tensor(variable_to_operator_edges)
        data["variable", "appears_in", "constraint"].edge_index = self.build_edge_index_tensor(variable_to_constraint_edges)
        data["operator", "connects_variable_to", "constraint"].edge_index = self.build_edge_index_tensor(operator_to_constraint_edges)
        data["meta", "has_constraint", "constraint"].edge_index = self.build_edge_index_tensor(meta_to_constraint_edges)
        if meta_connected_to_all:
            data["meta", "has_variable", "variable"].edge_index = self.build_edge_index_tensor([[0, i] for i in range(len(self.base_variables))])
            data["meta", "has_value", "value"].edge_index = self.build_edge_index_tensor([[0, i] for i in range(2)])
            data["meta", "has_operator", "operator"].edge_index = self.build_edge_index_tensor([[0, i] for i in range(len(operators))])
             
        # data = self.get_updated_heterodata(data)
        
        T.ToUndirected()(data)

        return data
    
    def get_updated_heterodata(self, data):
        node_id_mapping = create_node_id_mapping(data)
        homogeneous = hetero_data_to_homogeneous(data, node_id_mapping)

        centrality_measures = self.calculate_centrality_measures(homogeneous)  # Calculate centrality measures for all nodes
        value_centrality = []
        variable_centrality = []
        constraint_centrality = []
        meta_centrality = []
        operator_centrality = []

        # Fill the lists with the corresponding centrality measures
        for node_id, node_type_and_idx in node_id_mapping.items():
            node_type, node_idx = node_type_and_idx
            node_centrality = centrality_measures[node_id, :]

            if node_type == "value":
                value_centrality.append((node_idx, node_centrality))
            elif node_type == "variable":
                variable_centrality.append((node_idx, node_centrality))
            elif node_type == "constraint":
                constraint_centrality.append((node_idx, node_centrality))
            elif node_type == "meta":
                meta_centrality.append((node_idx, node_centrality))
            elif node_type == "operator":
                operator_centrality.append((node_idx, node_centrality))

        # Sort the lists based on the node index within its type
        value_centrality.sort(key=lambda x: x[0])
        variable_centrality.sort(key=lambda x: x[0])
        constraint_centrality.sort(key=lambda x: x[0])
        meta_centrality.sort(key=lambda x: x[0])
        operator_centrality.sort(key=lambda x: x[0])

        # Extract only the centrality measures from the sorted lists
        value_centrality = np.array([x[1] for x in value_centrality])
        variable_centrality = np.array([x[1] for x in variable_centrality])
        constraint_centrality = np.array([x[1] for x in constraint_centrality])
        meta_centrality = np.array([x[1] for x in meta_centrality])
        operator_centrality = np.array([x[1] for x in operator_centrality])

        data["variable"].x = torch.tensor(variable_centrality, dtype=torch.float32)
        label = 1 if self.is_sat else 0
        data["variable"].y = torch.tensor([label])
        data["constraint"].x = torch.tensor(constraint_centrality, dtype=torch.float32)

        return data
    
    def calculate_centrality_measures(self, homogeneous_graph):
        betw_cent = betweenness_centrality(homogeneous_graph)
        eigv_cent = nx.eigenvector_centrality_numpy(homogeneous_graph)
        close_cent = closeness_centrality(homogeneous_graph)

        # Normalize centrality measures
        betw_cent = np.array([betw_cent[i] for i in sorted(betw_cent.keys())])
        eigv_cent = np.array([eigv_cent[i] for i in sorted(eigv_cent.keys())])
        close_cent = np.array([close_cent[i] for i in sorted(close_cent.keys())])

        return np.vstack((betw_cent, eigv_cent, close_cent)).T

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


def create_node_id_mapping(data):
    node_id_mapping = {}
    current_id = 0

    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        for i in range(num_nodes):
            node_id_mapping[current_id] = (node_type, i)
            current_id += 1

    return node_id_mapping


def hetero_data_to_homogeneous(data, node_id_mapping):
    G = nx.Graph()

    for node_id in node_id_mapping:
        G.add_node(node_id)

    for edge_type in data.edge_types:
        src, dst = data[edge_type].edge_index
        src = src.numpy()
        dst = dst.numpy()
        edges = zip(src, dst)

        for src_node, dst_node in edges:
            src_unique_id = [key for key, value in node_id_mapping.items() if value == (edge_type[0], src_node)][0]
            dst_unique_id = [key for key, value in node_id_mapping.items() if value == (edge_type[2], dst_node)][0]

            G.add_edge(src_unique_id, dst_unique_id)

    return G