import torch
from typing import List
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


class GraphColoringInstance:
    def __init__(self, filename, color_embeddings):
        self.filename = filename
        self.color_embeddings = color_embeddings
    
    def get_gc_specific_representation(self):
        with open(self.filename, "r") as f:
            lines = f.readlines()

        positive_instance, negative_instance = HeteroData(), HeteroData()
        n_vertices = int(lines[1].split(':')[1])
        chromatic_number = int(lines[-2].strip())
        frozen_edge = torch.tensor([[int(i)] for i in lines[-4].strip().split()])

        vertices = torch.ones(n_vertices, 1)
        positive_instance["vertex"].x = vertices
        negative_instance["vertex"].x = vertices
        colors = self.color_embeddings[:chromatic_number]       
        positive_instance["color"].x = colors
        negative_instance["color"].x = colors
        vertex_to_vertex_edge_list = []
        for line in lines[6:]:
            if line.strip() == '-1':
                break
            vertex_to_vertex_edge_list.append(list(map(int, line.strip().split())))
        
        vertex_to_colors_edge_index = self.build_edge_index_tensor(
            torch.tensor(
                    [[i, j] for i in range(n_vertices) for j in range(chromatic_number)]
                )
            )

        positive_vertex_to_vertex_edge_index = self.build_edge_index_tensor(vertex_to_vertex_edge_list)
        positive_instance["vertex", "connected_to", "vertex"].edge_index = positive_vertex_to_vertex_edge_index
        positive_instance["vertex", "connected_to", "color"].edge_index = vertex_to_colors_edge_index
        positive_instance.label = torch.tensor([1])
        positive_instance.filename = self.filename
        T.ToUndirected()(positive_instance)

        negative_vertex_to_vertex_edge_index = torch.cat([positive_vertex_to_vertex_edge_index, frozen_edge], dim=1)
        negative_instance["vertex", "connected_to", "vertex"].edge_index = negative_vertex_to_vertex_edge_index        
        negative_instance["vertex", "connected_to", "color"].edge_index = vertex_to_colors_edge_index        
        negative_instance.label = torch.tensor([0])
        negative_instance.filename = self.filename
        T.ToUndirected()(negative_instance)
        
        return positive_instance, negative_instance

    def build_edge_index_tensor(self, edges:List)->torch.Tensor:
        return torch.Tensor(edges).long().t().contiguous()
