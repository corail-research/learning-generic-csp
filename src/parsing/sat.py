from typing import List
import torch
from torch_geometric.data import HeteroData


def parse_dimacs_cnf(filepath:str):
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    problem_definition = lines[0].strip().split()
    num_literals = problem_definition[2]
    num_clauses = problem_definition[3]
    
    clauses = []
    for i in range(1, len(lines)):
        line = lines[i].strip().split()[:-1]
        new_clause = Clause(line)
        clauses.append(new_clause)
    
    return CNF(clauses)

class Clause:
    """
    Simple class representing an OR clause; i.e. a line in a CNF problem.
    """
    def __init__(self, line:List[str]):
        self.variables = {int(var) for var in line}

class CNF:
    def __init__(self, clauses:List[Clause]):
        self.clauses = clauses
        variables = set()
        for clause in self.clauses:
            base_vars = {abs(var) for var in clause.variables}
            variables = variables.union(base_vars)
        variables = list(variables)
        self.variables = sorted(variables)
    
    def build_heterogeneous_graph(self):
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
        data["value"].x = torch.Tensor([[0], [1]])
        data["operator"].x = torch.Tensor([[1] for _ in range(negation_operator_id)])
        data["constraint"].x = torch.Tensor(constraints)

        data["variable", "connected_to", "value"].edge_index = torch.Tensor(variable_to_value_edges).long()
        data["variable", "connected_to", "operator"].edge_index = torch.Tensor(variable_to_operator_edges).long()
        data["variable", "connected_to", "constraint"].edge_index = torch.Tensor(variable_to_constraint_edges).long()
        data["operator", "connected_to", "constraint"].edge_index = torch.Tensor(operator_to_constraint_edges).long()
        data["constraint", "connected_to", "constraint"].edge_index = torch.Tensor(constraint_to_constraint_edges).long()

        return data

    def get_sat_variable_to_domain_edges(self, variables):
        edges = []
        for variable in variables:
            for value in [0, 1]:
                new_edge = [variable, value]
                edges.append(new_edge)
        return edges

if __name__ == "__main__":
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.datasets import OGB_MAG
    from torch_geometric.nn import SAGEConv, to_hetero


    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic Graph Representation\neurosat\dimacs\test\sr5\grp1\sr_n=0006_pk2=0.30_pg=0.40_t=9_sat=0.dimacs"
    cnf = parse_dimacs_cnf(test_path)
    data = cnf.build_heterogeneous_graph()

    class GNN(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x
    
    dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
    test = dataset[0]

    model = GNN(hidden_channels=64, out_channels=1)
    # model = to_hetero(model, test.metadata(), aggr='sum')
    # meta = {
    #     ('variable', 'can_take_value', 'value'): {'edge_type': '1'},
    #     ('variable', 'affected_by_operator', 'operator'): {'edge_type': '2'},
    #     ('variable', 'is_in', 'constraint'): {'edge_type': '3'},
    #     ('operator', 'connected_to', 'constraint'): {'edge_type': '4'},
    #     ('constraint', 'related_to', 'constraint'): {'edge_type': '5'},
    # }
    # for key, value in meta.items():
    #     data[key].metadata = value
    
    model = to_hetero(model, data.metadata(), aggr='sum', debug=True)
