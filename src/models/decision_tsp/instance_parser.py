import torch
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
                    self.optimal_value = self.compute_tour_cost(self.optimal_tour)
                i += 1

    def compute_tour_cost(self, tour, is_open=False):
        num_cities = len(tour)
        total_cost = 0.0

        # Calculate the total distance by summing up the distances between consecutive cities
        for i in range(num_cities - 1):
            from_city = tour[i]
            to_city = tour[i + 1]
            total_cost += self.distance_matrix[from_city][to_city]

        # Add the distance from the last city back to the starting city for closed tours
        if not is_open:
            total_cost += self.distance_matrix[tour[-1]][tour[0]]

        return total_cost
    
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
                target_cost_positive = self.optimal_value + target_deviation * self.optimal_value
                target_cost_negative = self.optimal_value - target_deviation * self.optimal_value
                arc_features_positive.append([self.distance_matrix[i][j], target_cost_positive])
                arc_features_negative.append([self.distance_matrix[i][j], target_cost_negative])
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
        data_positive.filename = self.filename
        data_positive.label = 0
        T.ToUndirected()(data_negative)

        return data_positive, data_negative
    
    def build_edge_index_tensor(self, edges:List)->torch.Tensor:
        return torch.Tensor(edges).long().t().contiguous()

if __name__ == "__main__":
    instance = TSPInstance('data/0.graph')
    instance.get_dtsp_specific_representation(0.02)
    a=1