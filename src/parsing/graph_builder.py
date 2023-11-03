from instance import XCSP3Instance

import networkx as nx
from typing import List
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt


class XCSP3GraphBuilder:
    def __init__(self, instance: XCSP3Instance):
        self.instance = instance
        self.domain_union = self.instance.variables.domain_union
        print(self.domain_union)
        self.variable_types = self.instance.variables.variable_types
        self.variable_to_operator_edges = []
        self.variable_to_value_edges = []
        self.variable_to_constraint_edges = []
        self.value_to_operator_edges = []
        self.operator_to_operator_edges = []
        self.operator_to_constraint_edges = []
        self.constraint_to_meta_edges = []
        self.constraint_features = []
        self.operator_features = []
        self.variable_features = []
        self.value_features = []
        self.meta_features = []
        self.operator_type_ids = SubTypeIDManager("operators")
        self.variable_type_ids = SubTypeIDManager("variables")
        self.value_type_ids = SubTypeIDManager("values")
        self.constraint_type_ids = SubTypeIDManager("self.constraint_features")
    
    def build_graph(self) -> HeteroData:
        """Builds a heterogeneous graph representation of the instance
        Returns:
            data (torch_geometric.data.HeteroData): graph for the SAT problem
        
        Edge Types:
        - value_to_operator: value -> operator (for combo operators used in extension self.constraint_features)
        - variable_to_value: variable -> value
        - variable_to_constraint: variable -> constraint
        - variable_to_operator: variable -> operator        
        - operator_to_constraint: operator -> constraint
        - operator_to_operator: constraint -> operator
        - constraint_to_meta: constraint -> meta    
        """
        int_value_subtype_dict = self.value_type_ids.add_subtype_id("int")
        real_value_subtype_dict = self.value_type_ids.add_subtype_id("real")
        for value in self.domain_union:
            value_type = "int" if isinstance(value, int) else "real"
            self.value_type_ids.add_node_id(subtype_name=value_type, name=value) # id of the value node
            if len(self.domain_union) > 1:
                current_value_features = [1, 0] if value_type == "int" else [0, value]
            else:
                current_value_features = [1] if value_type == "int" else [value]
            self.value_features.append(current_value_features)


        base_variables = self.instance.get_all_variables()
        
        int_var_subtype_dict = self.variable_type_ids.add_subtype_id("int")
        real_var_subtype_dict = self.variable_type_ids.add_subtype_id("real")

        for var_name, var in base_variables.items():
            variable_type = var.variable_type
            self.variable_type_ids.add_node_id(subtype_name=variable_type, name=var_name) # type of the variable?
            if len(self.variable_types) > 1:
                current_variable_features = [1, 0] if variable_type == "int" else [0, 1]
            else:
                current_variable_features = [1]
            self.variable_features.append(current_variable_features)

        for variable_name, variable in base_variables.items():
            for value in variable.domain:
                variable_id = self.variable_type_ids.get_node_id(name=variable_name)
                value_id = self.value_type_ids.get_node_id(name=value)
                new_pair = [variable_id, value_id]
                self.variable_to_value_edges.append(new_pair)
        
        for constraint_type, constraints in self.instance.constraints.items():
            if constraint_type == "extension":
                for constraint in constraints:
                    self.add_extension_constraint_to_graph(constraint)
            elif constraint_type == "sum":
                pass
            elif constraint_type == "allDifferent":
                for constraint in constraints:
                    self.add_all_different_constraint_to_graph(constraint)
            elif constraint_type == "intension":
                pass
            elif constraint_type == "element":
                pass
        
        self.operator_features = self.one_hot_encode(self.operator_features)
        self.constraint_features = self.one_hot_encode(self.constraint_features)

        data = HeteroData()

        
        return data
    
    def get_variable_to_value_edges(self)->List:
        pass

    def add_extension_constraint_to_graph(self, constraint):
        if "supports" in constraint:
            constraint_subtype_name = "extension+"
            to_parse = "supports"

        elif "conflicts" in constraint:
            constraint_subtype_name = "extension-"
            to_parse = "conflicts"

        constraint_subtype_dict = self.constraint_type_ids.add_subtype_id(constraint_subtype_name)
        extension_constraint_id = self.constraint_type_ids.add_node_id(constraint_subtype_name)
        self.constraint_features.append(constraint_subtype_dict["id"])

        combo_subtype_id = self.operator_type_ids.add_subtype_id("combo")["id"]
        ext_tuple_subtype_id = self.operator_type_ids.add_subtype_id("extension_tuple")["id"]

        constraint_variables = constraint["variables"]
        tuple_elements_to_parse = constraint[to_parse]

        for tuple_element in tuple_elements_to_parse:
            tuple_id = self.operator_type_ids.add_node_id(subtype_name="extension_tuple")
            self.operator_features.append(ext_tuple_subtype_id)
            
            for i in range(len(tuple_element)):
                combo_name = "combo_" + str(tuple_element[i])+ constraint_variables[i]
                if tuple_element[i] is None: # this happens when the support is *, converted to None by the parser
                    continue
                combo_id = self.operator_type_ids.get_node_id(name=combo_name)
                if not combo_id:
                    combo_id = self.operator_type_ids.add_node_id(subtype_name="combo", name=combo_name) # id of the combo operator node
                
                variable_id = self.variable_type_ids.get_node_id(name=constraint_variables[i])
                domain_value_id = self.value_type_ids.get_node_id(name=tuple_element[i])
                new_var_op_pair = [variable_id, combo_id]
                self.variable_to_operator_edges.append(new_var_op_pair)
                new_val_op_pair = [domain_value_id, combo_id]
                self.value_to_operator_edges.append(new_val_op_pair)
                self.operator_to_operator_edges.append([combo_id, tuple_id])
                self.operator_features.append(combo_subtype_id)
            
            self.operator_to_constraint_edges.append([tuple_id, extension_constraint_id])
        # add edge between everry variable and the constraint
        for variable_name in constraint_variables:
            variable_id = self.variable_type_ids.get_node_id(name=variable_name)
            self.variable_to_constraint_edges.append([variable_id, extension_constraint_id])
        

    def add_all_different_constraint_to_graph(self, constraint):
        all_different_subtype_id = self.constraint_type_ids.add_subtype_id("allDifferent")["id"]
        all_different_id = self.constraint_type_ids.add_node_id("allDifferent")
        for variable_name in constraint:
            variable_id = self.variable_type_ids.get_node_id(name=variable_name)
            self.variable_to_constraint_edges.append([variable_id, all_different_id])
            self.constraint_features.append(all_different_subtype_id)
        return        

    def one_hot_encode(self, values):
        """
        Creates a one-hot encoding of a list of integers.
        """
        # Get the maximum value in the list to determine the length of the one-hot vectors
        max_value = max(values)
        # Create a one-hot encoded list for each integer
        one_hot_encoded = [[1 if i == value else 0 for i in range(max_value + 1)] for value in int_list]

        return one_hot_encoded

    def build_edge_index_tensor(self, edges:List)->torch.Tensor:
        return torch.Tensor(edges).long().t().contiguous()

class SubTypeIDManager:
    def __init__(self, name):
        self.name = name
        self.subtype_ids = {} # used as a lookup table for subtype ids within the given type (e.g. constraints: extension, sum, etc.)
        self.node_ids = {} # used as a lookup table for node ids within the given node type
        self.node_names = {} # used as a lookup table for node names given a node id 
        self.current_node_id = 0
        self.current_subtype_id = 0
    
    def get_node_id(self, name):
        return self.node_ids.get(name, None)
    
    def get_node_name(self, node_id):
        return self.node_names.get(node_id, None)

    def add_node_id(self, subtype_name, name=""):
        self.subtype_ids[subtype_name]["count"] = self.subtype_ids[subtype_name]["count"] + 1
        if name == "":
            count = self.subtype_ids[subtype_name]["count"]
            name  = subtype_name + f"{count}"
        if name not in self.node_ids:
            self.node_ids[name] = self.current_node_id
            self.node_names[self.current_node_id] = name
            self.current_node_id += 1
        return self.node_ids[name]
    
    def get_subtype_id(self, name):
        return self.subtype_ids.get(name, None)
    
    def add_subtype_id(self, name):
        if name not in self.subtype_ids:
            self.subtype_ids[name] = {"id": self.current_subtype_id, "count": 0}
            self.current_subtype_id += 1
        
        return self.subtype_ids[name]

if __name__ == "__main__":
    from instance import parse_instance
    import os

    # test_files_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCSP23"
    # files = [os.path.join(test_files_path, file) for file in os.listdir(test_files_path)]

    filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\decision_tsp\text.xml"
    instance = parse_instance(filepath)
    graph_builder = XCSP3GraphBuilder(instance)
    graph_builder.build_graph()
    def print_variable_to_operator_edges(builder):
        for edge in builder.variable_to_operator_edges:
            formatted_edge = [builder.variable_type_ids.get_node_name(edge[0]), builder.operator_type_ids.get_node_name(edge[1])]
            print(formatted_edge)

    def print_variable_to_value_edges(builder):
        for edge in builder.variable_to_value_edges:
            formatted_edge = [builder.variable_type_ids.get_node_name(edge[0]), builder.value_type_ids.get_node_name(edge[1])]
            print(formatted_edge)

    def print_variable_to_constraint_edges(builder):
        for edge in builder.variable_to_constraint_edges:
            formatted_edge = [builder.variable_type_ids.get_node_name(edge[0]), builder.constraint_type_ids.get_node_name(edge[1])]
            print(formatted_edge)

    def print_value_to_operator_edges(builder):
        for edge in builder.value_to_operator_edges:
            formatted_edge = [builder.value_type_ids.get_node_name(edge[0]), builder.operator_type_ids.get_node_name(edge[1])]
            print(formatted_edge)

    def print_operator_to_operator_edges(builder):
        for edge in builder.operator_to_operator_edges:
            formatted_edge = [builder.operator_type_ids.get_node_name(edge[0]), builder.operator_type_ids.get_node_name(edge[1])]
            print(formatted_edge)

    def print_operator_to_constraint_edges(builder):
        for edge in builder.operator_to_constraint_edges:
            formatted_edge = [builder.operator_type_ids.get_node_name(edge[0]), builder.constraint_type_ids.get_node_name(edge[1])]
            print(formatted_edge)

    print("\nvariable_to_value_edges\n")
    print_variable_to_value_edges(graph_builder)

    print("\nvariable_to_operator_edges\n")
    print_variable_to_operator_edges(graph_builder)

    print("\nvariable_to_constraint_edges\n")
    print_variable_to_constraint_edges(graph_builder)

    print("\nvalue_to_operator_edges\n")
    print_value_to_operator_edges(graph_builder)

    print("\noperator_to_operator_edges\n")
    print_operator_to_operator_edges(graph_builder)

    print("\noperator_to_constraint_edges\n")
    print_operator_to_constraint_edges(graph_builder)

    def print_constraint_features(builder):
        for feature in builder.constraint_features:
            print(f"Constraint feature: {feature}")

    def print_operator_features(builder):
        for feature in builder.operator_features:
            print(f"Operator feature: {feature}")

    def print_variable_features(builder):
        for feature in builder.variable_features:
            print(f"Variable feature: {feature}")

    def print_value_features(builder):
        for feature in builder.value_features:
            print(f"Value feature: {feature}")

    def print_meta_features(builder):
        for feature in builder.meta_features:
            print(f"Meta feature: {feature}")
    
    print("\nconstraint_features\n")
    print_constraint_features(graph_builder)
    print("\noperator_features\n")
    print_operator_features(graph_builder)
    print("\nvariable_features\n")
    print_variable_features(graph_builder)
    print("\nvalue_features\n")
    print_value_features(graph_builder)
    print("\nmeta_features\n")
    print_meta_features(graph_builder)
    
    a=1