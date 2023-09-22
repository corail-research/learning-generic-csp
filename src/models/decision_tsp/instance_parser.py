import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from typing import List


class TSPInstance:
    def __init__(self, filename):
        self.filename = filename
        self.distance_matrix = None
        self.connectivity = None
        self.optimal_tour = None
        self.parse_instance()

    def parse_instance(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith('DIMENSION'):
                    self.dimension = int(lines[i].split(':')[1].strip())
                elif lines[i].startswith('EDGE_WEIGHT_TYPE'):
                    self.edge_weight_type = lines[i].split(':')[1].strip()
                elif lines[i].startswith('EDGE_WEIGHT_FORMAT'):
                    self.edge_weight_format = lines[i].split(':')[1].strip()
                elif lines[i].startswith('EDGE_DATA_SECTION'):
                    i += 1
                    edges = []
                    while lines[i].strip() != '-1':
                        edges.append(list(map(int, lines[i].strip().split())))
                        i += 1
                    self.connectivity = len(edges) / (self.dimension * (self.dimension - 1) / 2)
                elif lines[i].startswith('EDGE_WEIGHT_SECTION'):
                    i += 1
                    self.distance_matrix = []
                    for j in range(self.dimension):
                        row = list(map(float, lines[i+j].strip().split()))
                        self.distance_matrix.append(row)
                elif lines[i].startswith('TOUR_SECTION'):
                    i += 1
                    self.optimal_tour = list(map(int, lines[i].strip().split()))
                elif lines[i].startswith("OPTIMAL_VALUE"):
                    i += 1
                    self.optimal_value = float(lines[i].strip())
                i += 1

    
    def get_dtsp_specific_representation(self, target_deviation):
        """Build the graph representation used in the Prates DTSP paper.
        This representation takes a TSP instance and builds a bipartite graph where node types are:
         - cities (nodes in the base instance)
         - arcs (edges in the base instance)
        The nodes are connected if an arc is connected to a city in the base graph
        Cities have a feature vector of 1s, while arcs have a feature vector of [w, C], where w is the edge weight and C is the target 
        cost for the decision TSP.

        Args:
            target_cost (float): the target deviation cost for the decision TSP, in percentage. For example, 0.02 means that the target cost is 2% more 
            or less than the optimal cost.
        
        Returns:
            data (tuple(torch_geometric.data.HeteroData)): graph for the DTSP problem, one for the positive instance (cost = optimal value + target_deviation * optimal_value)
             and one for the negative instance (cost = optimal value - target_deviation * optimal_value)
             labels are stored in data.label
        """
        target_cost_positive = self.optimal_value * (1 + target_deviation)
        target_cost_negative = self.optimal_value * (1 - target_deviation)
        # Nodes and node stuff
        cities = []
        arcs = []
        arc_features_positive = []
        arc_features_negative = []
        arcs_to_id_mapping = {}
        arc_counter = 0
        for i in range(self.dimension):
            cities.append([1])
            for j in range(i+1, self.dimension):
                arcs.append([i, arc_counter])
                arcs.append([j, arc_counter])
                distance = self.distance_matrix[i][j]
                arc_features_positive.append([distance, target_cost_positive])
                arc_features_negative.append([distance, target_cost_negative])
                arcs_to_id_mapping[(i, j)] = arc_counter
                arc_counter += 1

        data_positive, data_negative = HeteroData(), HeteroData()
        data_positive["city"].x = torch.Tensor(cities)
        data_positive["arc"].x = torch.Tensor(arc_features_positive)
        data_positive["city", "connected_to", "arc"].edge_index = self.build_edge_index_tensor(arcs)
        data_positive.filename = self.filename
        data_positive.label = 1
        T.ToUndirected()(data_positive)

        data_negative["city"].x = torch.Tensor(cities)
        data_negative["arc"].x = torch.Tensor(arc_features_negative)
        data_negative["city", "connected_to", "arc"].edge_index = self.build_edge_index_tensor(arcs)
        data_negative.filename = self.filename
        data_negative.label = 0
        T.ToUndirected()(data_negative)

        return data_positive, data_negative
    
    def get_circuit_based_representation(self, target_deviation):
        """
        CP Model for the Traveling Salesman Problem (TSP) with Circuit constraint

        Parameters:
            distances (matrix of floats): Distance matrix where distances[i][j] represents the distance between node i and node j.

        Variables:
            x (array of ints): Values are the successor node
            distance (float): total distance traveled. - OPTIONAL, not implemented

        Objective:
            Minimize the total distance traveled.

        Constraints:
            1. All nodes must be visited exactly once: AllDifferent(x)
            2. Circuit constraint: Circuit(x)
        
        Operators:
            1. Distances are represented as operators between

        Args:
            target_cost (float): the target deviation cost for the decision TSP, in percentage. For example, 0.02 means that the target cost is 2% more 
            or less than the optimal cost.
        
        Returns:
            data (tuple(torch_geometric.data.HeteroData)): graph for the DTSP problem, one for the positive instance (cost = optimal value + target_deviation * optimal_value)
             and one for the negative instance (cost = optimal value - target_deviation * optimal_value)
             labels are stored in data.label
        """
        target_cost_positive = self.optimal_value * (1 + target_deviation)
        target_cost_negative = self.optimal_value * (1 - target_deviation)

        x_features_positive = [[1, target_cost_positive] for _ in range(self.dimension)]
        x_features_negative = [[1, target_cost_negative] for _ in range(self.dimension)]
        values = [[1] for i in range(self.dimension)]
        constraints = [[1, 0], [0, 1]] # Circuit and AllDifferent Constraints
        x_to_constraint_edges = []

        for i in range(self.dimension):
            x_to_constraint_edges.append([i, 0])
            x_to_constraint_edges.append([i, 1])
        
        values_to_x_edges = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    values_to_x_edges.append([i, j])
        
        distance_parameters_features = []
        distance_parameter_to_x_edges = []
        distance_parameter_to_value_edges = []
        distance_parameter_counter = 0
    
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                distance = self.distance_matrix[i][j]
                distance_parameters_features.append([distance])
                distance_parameter_to_x_edges.append([distance_parameter_counter, i])
                distance_parameter_to_x_edges.append([distance_parameter_counter, j])
                distance_parameter_to_value_edges.append([distance_parameter_counter, i])
                distance_parameter_to_value_edges.append([distance_parameter_counter, j])
                distance_parameter_counter += 1
                

        data_positive, data_negative = HeteroData(), HeteroData()

        data_positive["x"].x = torch.Tensor(x_features_positive)
        data_positive["constraint"].x = torch.Tensor(constraints)
        data_positive["value"].x = torch.Tensor(values)
        data_positive["operator"].x = torch.Tensor(distance_parameters_features)
        data_positive["x", "involved_in", "constraint"].edge_index = self.build_edge_index_tensor(x_to_constraint_edges)
        data_positive["x", "has_value", "value"].edge_index = self.build_edge_index_tensor(values_to_x_edges)
        data_positive["operator", "has_parameter", "x"].edge_index = self.build_edge_index_tensor(distance_parameter_to_x_edges)
        data_positive["operator", "has_parameter", "value"].edge_index = self.build_edge_index_tensor(distance_parameter_to_value_edges)
        data_positive.filename = self.filename
        data_positive.label = 1
        T.ToUndirected()(data_positive)

        data_negative["x"].x = torch.Tensor(x_features_negative)
        data_negative["constraint"].x = torch.Tensor(constraints)
        data_negative["value"].x = torch.Tensor(values)
        data_negative["operator"].x = torch.Tensor(distance_parameters_features)
        data_negative["x", "involved_in", "constraint"].edge_index = self.build_edge_index_tensor(x_to_constraint_edges)
        data_negative["x", "has_value", "value"].edge_index = self.build_edge_index_tensor(values_to_x_edges)
        data_negative["operator", "has_parameter", "x"].edge_index = self.build_edge_index_tensor(distance_parameter_to_x_edges)
        data_negative["operator", "has_parameter", "value"].edge_index = self.build_edge_index_tensor(distance_parameter_to_value_edges)
        data_negative.filename = self.filename
        data_negative.label = 0

        T.ToUndirected()(data_positive)
        T.ToUndirected()(data_negative)

        return data_positive, data_negative
    
    def build_edge_index_tensor(self, edges:List)->torch.Tensor:
        return torch.Tensor(edges).long().t().contiguous()

if __name__ == "__main__":
    instance = TSPInstance('data/0.graph')
    instance.get_dtsp_specific_representation(0.02)
    a=1