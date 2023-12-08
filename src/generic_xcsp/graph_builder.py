try:
    from generic_xcsp.instance import XCSP3Instance
except:
    from instance import XCSP3Instance
from typing import List
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


class XCSP3GraphBuilder:
    def __init__(self, instance: XCSP3Instance, filename: str):
        self.instance = instance
        self.domain_union = self.instance.variables.domain_union
        self.variable_types = self.instance.variables.variable_types
        self.variable_to_operator_edges = []
        self.variable_to_value_edges = []
        self.variable_to_constraint_edges = []
        self.variable_to_objective_edges = []
        self.variable_to_variable_edges = []
        self.value_to_operator_edges = []
        self.operator_to_operator_edges = []
        self.operator_to_constraint_edges = []
        self.operator_to_objective_edges = []
        self.constraint_to_objective_edges = []
        self.constraint_features = []
        self.operator_features = []
        self.variable_features = []
        self.value_features = []
        self.objective_features = []
        self.operator_node_values = {} # map the id of operators to their desired values
        self.operator_type_ids = SubTypeIDManager("operators")
        self.variable_type_ids = SubTypeIDManager("variables")
        self.value_type_ids = SubTypeIDManager("values")
        self.constraint_type_ids = SubTypeIDManager("self.constraint_features")
        self.filename = filename

    def get_marty_et_al_graph_representation(self):
        """Builds a heterogeneous graph representation based on the "generic" representation of Marty et al.
        Returns:
            data (torch_geometric.data.HeteroData): graph for the SAT problem
        
        Edge Types:
        - value_to_operator: value -> operator (for combo operators used in extension self.constraint_features)
        - variable_to_value: variable -> value
        - variable_to_constraint: variable -> constraint
        """
        int_value_subtype_dict = self.value_type_ids.add_subtype_id("int")
        real_value_subtype_dict = self.value_type_ids.add_subtype_id("real")
        for value in self.domain_union:
            value_type = "int" if isinstance(value, int) else "real"
            self.value_type_ids.add_node_id(subtype_name=value_type, name=value)
            current_value_features = [value]
            self.value_features.append(current_value_features)
        
        base_variables = self.instance.get_all_variables()
        
        var_subtype_dict = self.variable_type_ids.add_subtype_id("var")
       
        for var_name, var in base_variables.items():
            variable_type = "var"
            self.variable_type_ids.add_node_id(subtype_name=variable_type, name=var_name) # type of the variable?
            current_variable_features = [len(var.domain), 0]
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
                    self.add_extension_marty_et_al(constraint)
            elif constraint_type == "sum":
                for constraint in constraints:
                    self.add_sum_marty_et_al(constraint)
            elif constraint_type == "allDifferent":
                for constraint in constraints:
                    self.add_all_different_marty_et_al(constraint)
            elif constraint_type == "intension":
                for constraint in constraints:
                    self.add_intension_marty_et_al(constraint)
            elif constraint_type == "element":
                for constraint in constraints:
                    self.add_element_marty_et_al(constraint)
        
        self.constraint_features = self.one_hot_encode(self.constraint_features)
        if self.instance.optimal_deviation_factor is not None:
            data_positive, data_negative = self.build_positive_and_negative_pair()
            data_positive.filename = self.filename
            data_negative.filename = self.filename

            return data_positive, data_negative
        else:
            data = self.build_graph()
            data.filename = self.filename
        return data

    def get_graph_representation(self) -> HeteroData:
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
            if len(self.variable_types) > 1:
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
                for constraint in constraints:
                    self.add_sum_constraint_to_graph(constraint)
            elif constraint_type == "allDifferent":
                for constraint in constraints:
                    self.add_all_different_constraint_to_graph(constraint)
            elif constraint_type == "intension":
                for constraint in constraints:
                    self.add_intension_constraint_to_graph(constraint)
            elif constraint_type == "element":
                for constraint in constraints:
                    self.add_element_constraint_to_graph(constraint)

        self.add_objective_to_graph()
        if self.operator_features:
            self.operator_features = self.one_hot_encode(self.operator_features, self.operator_node_values)
        self.constraint_features = self.one_hot_encode(self.constraint_features)

        if self.instance.optimal_deviation_factor is not None:
            data_positive, data_negative = self.build_positive_and_negative_pair()
            data_positive.filename = self.filename
            data_negative.filename = self.filename

            return data_positive, data_negative
        else:
            data = self.build_graph()
            data.filename = self.filename
        return data
    
    def build_positive_and_negative_pair(self):
        if self.instance.optimal_deviation_factor:
            if self.instance.objective.minimize:
                objective_features_positive = self.instance.optimal_value * (1 + self.instance.optimal_deviation_factor) 
                objective_features_negative = self.instance.optimal_value * (1 - self.instance.optimal_deviation_factor)
            else:
                objective_features_positive = self.instance.optimal_value * (1 - self.instance.optimal_deviation_factor) 
                objective_features_negative = self.instance.optimal_value * (1 + self.instance.optimal_deviation_factor)

        elif self.instance.optimal_deviation_difference:
            if self.instance.objective.minimize:
                objective_features_positive = self.instance.optimal_value + self.instance.optimal_deviation_difference
                objective_features_negative = self.instance.optimal_value - self.instance.optimal_deviation_difference
            else:
                objective_features_positive = self.instance.optimal_value - self.instance.optimal_deviation_difference
                objective_features_negative = self.instance.optimal_value + self.instance.optimal_deviation_difference
            

        data_positive, data_negative = HeteroData(), HeteroData()
        # Positive sample
        if self.variable_features:
            data_positive["variable"].x = torch.Tensor(self.variable_features)
            data_negative["variable"].x = torch.Tensor(self.variable_features)
        if self.value_features:
            data_positive["value"].x = torch.Tensor(self.value_features)
            data_negative["value"].x = torch.Tensor(self.value_features)
        if self.operator_features:
            data_positive["operator"].x = torch.Tensor(self.operator_features)
            data_negative["operator"].x = torch.Tensor(self.operator_features)
        if self.constraint_features:
            data_positive["constraint"].x = torch.Tensor(self.constraint_features)
            data_negative["constraint"].x = torch.Tensor(self.constraint_features)
        #if self.objective_features:
        data_positive["objective"].x = torch.Tensor([objective_features_positive]).unsqueeze(0)
        data_negative["objective"].x = torch.Tensor([objective_features_negative]).unsqueeze(0)

        data_positive["variable", "connected_to", "value"].edge_index = self.build_edge_index_tensor(self.variable_to_value_edges)
        data_negative["variable", "connected_to", "value"].edge_index = self.build_edge_index_tensor(self.variable_to_value_edges)
        if self.variable_to_operator_edges:
            data_positive["variable", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.variable_to_operator_edges)
            data_negative["variable", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.variable_to_operator_edges)
        if self.variable_to_constraint_edges:
            data_positive["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.variable_to_constraint_edges)
            data_negative["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.variable_to_constraint_edges)
        if self.operator_to_operator_edges:    
            data_positive["operator", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.operator_to_operator_edges)
            data_negative["operator", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.operator_to_operator_edges)
        if self.operator_to_constraint_edges:
            data_positive["operator", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.operator_to_constraint_edges)
            data_negative["operator", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.operator_to_constraint_edges)
        if self.operator_to_objective_edges:
            data_positive["operator", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.operator_to_objective_edges)
            data_negative["operator", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.operator_to_objective_edges)
        if self.value_to_operator_edges:
            data_positive["value", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.value_to_operator_edges)
            data_negative["value", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.value_to_operator_edges)
        if self.variable_to_objective_edges:
            data_positive["variable", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.variable_to_objective_edges)
            data_negative["variable", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.variable_to_objective_edges)
        if self.constraint_to_objective_edges:
            data_positive["constraint", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.constraint_to_objective_edges)
            data_negative["constraint", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.constraint_to_objective_edges)
        
        data_positive.label = 1
        T.ToUndirected()(data_positive)
        data_negative.label = 0
        T.ToUndirected()(data_negative)

        return data_positive, data_negative

    def build_graph(self):
        data = HeteroData()
        # Positive sample
        data["variable"].x = torch.Tensor(self.variable_features)
        data["value"].x = torch.Tensor(self.value_features)
        if self.operator_features:
            data["operator"].x = torch.Tensor(self.operator_features)
        if self.constraint_features:
            data["constraint"].x = torch.Tensor(self.constraint_features)
        if self.instance.optimal_deviation_factor:
            objective_features = self.instance.optimal_value * (1 + self.instance.optimal_deviation_factor) 
        elif self.instance.optimal_deviation_difference:
            objective_features = self.instance.optimal_value + self.instance.optimal_deviation_difference
        else:
            objective_features = 1
        if self.objective_features:
            data["objective"].x = torch.Tensor([objective_features]).unsqueeze(0)

        data["variable", "connected_to", "value"].edge_index = self.build_edge_index_tensor(self.variable_to_value_edges)
        if self.variable_to_operator_edges:
            data["variable", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.variable_to_operator_edges)
        if self.variable_to_constraint_edges:
            data["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.variable_to_constraint_edges)
        if self.operator_to_operator_edges:    
            data["operator", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.operator_to_operator_edges)
        if self.operator_to_constraint_edges:
            data["operator", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.operator_to_constraint_edges)
        if self.operator_to_objective_edges:
            data["operator", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.operator_to_objective_edges)
        if self.value_to_operator_edges:
            data["value", "connected_to", "operator"].edge_index = self.build_edge_index_tensor(self.value_to_operator_edges)
        if self.variable_to_objective_edges:
            data["variable", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.variable_to_objective_edges)
        if self.constraint_to_objective_edges:
            data["constraint", "connected_to", "objective"].edge_index = self.build_edge_index_tensor(self.constraint_to_objective_edges)
        data.label = self.instance.objective.label
        T.ToUndirected()(data)
        
        return data

    def add_extension_marty_et_al(self, constraint):
        if "supports" in constraint:
            constraint_subtype_name = "extension+"

        elif "conflicts" in constraint:
            constraint_subtype_name = "extension-"

        constraint_subtype_dict = self.constraint_type_ids.add_subtype_id(constraint_subtype_name)
        extension_constraint_id = self.constraint_type_ids.add_node_id(constraint_subtype_name)
        self.constraint_features.append(constraint_subtype_dict["id"])

        constraint_variables = constraint["variables"]

        for i in range(len(constraint_variables)):
            variable_id = self.variable_type_ids.get_node_id(name=constraint_variables[i])
            self.variable_to_constraint_edges.append([variable_id, extension_constraint_id])
    
    def add_sum_marty_et_al(self, constraint):
        variables = constraint["variables"]
        
        sum_subtype_id = self.constraint_type_ids.add_subtype_id("sum")["id"]
        self.constraint_features.append(sum_subtype_id)
        sum_id = self.constraint_type_ids.add_node_id("sum")
        
        condition = constraint["condition"]
        condition_operand = condition["operand"]
        
        if not condition_operand.isdigit(): # in this case, the sum is compared to a variable
            condition_operand_id = self.variable_type_ids.get_node_id(name=condition_operand)
            self.variable_to_constraint_edges.append([condition_operand_id, sum_id])
        
        for variable in variables:
            variable_id = self.variable_type_ids.get_node_id(name=variable)
            self.variable_to_constraint_edges.append([variable_id, sum_id])
            
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
        
        self.constraint_to_objective_edges.append([extension_constraint_id, 0])
        
    def add_all_different_marty_et_al(self, constraint):
        all_different_subtype_id = self.constraint_type_ids.add_subtype_id("allDifferent")["id"]
        all_different_id = self.constraint_type_ids.add_node_id("allDifferent")
        for variable_name in constraint:
            variable_id = self.variable_type_ids.get_node_id(name=variable_name)
            self.variable_to_constraint_edges.append([variable_id, all_different_id])
            self.constraint_features.append(all_different_subtype_id)

    def add_intension_marty_et_al(self, constraint):
        # TODO: extend to other, non-graph coloring problems
        intension_subtype_id = self.constraint_type_ids.add_subtype_id("intension")["id"]
        intension_id = self.constraint_type_ids.add_node_id("intension")
        self.constraint_features.append(intension_subtype_id)
        for variable_name in constraint.children:
            variable_id = self.variable_type_ids.get_node_id(name=variable_name)
            self.variable_to_constraint_edges.append([variable_id, intension_id])
    
    def add_element_marty_et_al(self, constraint):
        index = constraint["index"]
        value = constraint["value"]
        element_list = constraint["list"]
        element_subtype_id = self.constraint_type_ids.add_subtype_id("element")["id"]
        element_id = self.constraint_type_ids.add_node_id("element")
        self.constraint_features.append(element_subtype_id)

        for i, current_element in enumerate(element_list):
            if type(current_element) == float or type(current_element) == int:
                pass
            else:
                current_element_id = self.variable_type_ids.get_node_id(name=current_element)
                self.variable_to_constraint_edges.append([current_element_id, element_id])
        
        if type(index) == float or type(index) == int:
            pass
        else:
            index_id = self.variable_type_ids.get_node_id(name=index)
            self.variable_to_constraint_edges.append([index_id, element_id])
        
        if type(value) == float or type(value) == int:
            pass
        else:
            value_id = self.variable_type_ids.get_node_id(name=value)
            self.variable_to_operato_edges.append([value_id, element_id])
        
    def add_all_different_constraint_to_graph(self, constraint):
        all_different_subtype_id = self.constraint_type_ids.add_subtype_id("allDifferent")["id"]
        all_different_id = self.constraint_type_ids.add_node_id("allDifferent")
        for variable_name in constraint:
            variable_id = self.variable_type_ids.get_node_id(name=variable_name)
            self.variable_to_constraint_edges.append([variable_id, all_different_id])
            self.constraint_features.append(all_different_subtype_id)
        
        self.constraint_to_objective_edges.append([all_different_id, 0])
    
    def add_objective_to_graph(self):
        
        if self.instance.objective.coeffs:
            multiply_subtype_id = self.operator_type_ids.add_subtype_id("multiply")["id"]
        else:
            multiply_subtype_id = None
        for i, variable in enumerate(self.instance.objective.variables):
            variable_id = self.variable_type_ids.get_node_id(name=variable)
            self.variable_to_objective_edges.append([variable_id, 0])
            if multiply_subtype_id:
                coeff = self.instance.objective.coeffs[i]
                self.add_coeff_to_objective_subgraph(variable_id, coeff, multiply_subtype_id)

    def add_coeff_to_objective_subgraph(self, variable_id, coeff, multiply_subtype_id):
        new_multiply_node_id = self.operator_type_ids.add_node_id(subtype_name="multiply")
        self.operator_features.append(multiply_subtype_id)    
        new_var_op_pair = [variable_id, new_multiply_node_id]
        self.variable_to_operator_edges.append(new_var_op_pair)
        self.operator_to_objective_edges.append([new_multiply_node_id, 0])
        self.operator_node_values[new_multiply_node_id] = float(coeff)
        
    def get_knapsack_specific_graph_representation(self):
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
        base_variables = self.instance.get_all_variables()
        
        int_var_subtype_dict = self.variable_type_ids.add_subtype_id("int")

        for var_name, var in base_variables.items():
            self.variable_type_ids.add_node_id(subtype_name="int", name=var_name) # type of the variable?
            positive_variable_features = [1]
            self.variable_features.append(positive_variable_features)

            negative_variable_name = "neg_" + var_name
            self.variable_type_ids.add_node_id(subtype_name="int", name=negative_variable_name) # type of the variable
            negative_variable_features = [1]
            self.variable_features.append(negative_variable_features)
            self.variable_to_variable_edges.append(
                [
                    self.variable_type_ids.get_node_id(name=var_name),
                    self.variable_type_ids.get_node_id(name=negative_variable_name)
                ]
            )

        for constraint_type, constraints in self.instance.constraints.items():
            for constraint in constraints:
                constraint_weights = self.add_knapsack_specific_sum_constraint_to_graph(constraint)
        
        objective_value_positive = self.instance.optimal_value * (1.02) 
        objective_value_negative = self.instance.optimal_value * (0.98)

        objective_weights_positive, positive_variable_to_constraint_edges = self.add_knapsack_specific_objective_to_graph(objective_value_positive)
        objective_weights_negative, negative_variable_to_constraint_edges = self.add_knapsack_specific_objective_to_graph(objective_value_negative)
                    
        data_positive, data_negative = HeteroData(), HeteroData()

        if self.variable_features:
            data_positive["variable"].x = torch.Tensor(self.variable_features)
            data_negative["variable"].x = torch.Tensor(self.variable_features)
        
        data_positive["constraint"].x = torch.Tensor([[1], [1]])
        data_negative["constraint"].x = torch.Tensor([[1], [1]])

        if self.variable_to_constraint_edges:
            data_positive["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.variable_to_constraint_edges + positive_variable_to_constraint_edges)
            data_negative["variable", "connected_to", "constraint"].edge_index = self.build_edge_index_tensor(self.variable_to_constraint_edges + negative_variable_to_constraint_edges)
        if self.variable_to_variable_edges:
            data_positive["variable", "connected_to", "variable"].edge_index = self.build_edge_index_tensor(self.variable_to_variable_edges)
            data_negative["variable", "connected_to", "variable"].edge_index = self.build_edge_index_tensor(self.variable_to_variable_edges)
        
        data_positive.label = 1
        T.ToUndirected()(data_positive)

        data_negative.label = 0
        T.ToUndirected()(data_negative)
        
        data_positive.filename = self.filename
        data_positive.variable_weights = torch.Tensor(constraint_weights + objective_weights_positive)
        data_negative.filename = self.filename
        data_negative.variable_weights = torch.Tensor(constraint_weights + objective_weights_negative)

        return data_positive, data_negative
    
    def add_knapsack_specific_sum_constraint_to_graph(self, constraint):
        variables = constraint["variables"]
        coeffs = constraint["coeffs"]
        
        sum_subtype_id = self.constraint_type_ids.add_subtype_id("sum")["id"]
        sum_id = self.constraint_type_ids.get_node_id(name="sum")
        if sum_id is None:
            sum_id = self.constraint_type_ids.add_node_id(subtype_name="sum", name="sum")
        
        condition = constraint["condition"]
        condition_operand = condition["operand"]
        sum_constraint_weights = []

        for i, variable in enumerate(variables):
            negative_variable_name = "neg_" + variable
            variable_id = self.variable_type_ids.get_node_id(name=negative_variable_name)
            corrected_weight = coeffs[i] * -1/float(condition_operand)
            sum_constraint_weights.append(corrected_weight)

            self.variable_to_constraint_edges.append([variable_id, sum_id])
        
        return sum_constraint_weights
    
    def add_knapsack_specific_objective_to_graph(self, optimal_value):
        sum_subtype_id = self.constraint_type_ids.add_subtype_id("sum")["id"]
        sum_id = self.constraint_type_ids.get_node_id(name="objective")
        if sum_id is None:
            sum_id = self.constraint_type_ids.add_node_id(subtype_name="sum", name="objective")
        weights = []
        new_variable_to_constraint_edges = []
        for i, variable in enumerate(self.instance.objective.variables):
            variable_id = self.variable_type_ids.get_node_id(name=variable)
            new_variable_to_constraint_edges.append([variable_id, sum_id])
            coeff = self.instance.objective.coeffs[i]
            corrected_weight = coeff/optimal_value
            weights.append(corrected_weight)

        return weights, new_variable_to_constraint_edges
    
    def add_sum_constraint_to_graph(self, constraint):
        variables = constraint["variables"]
        coeffs = constraint["coeffs"]
        
        sum_subtype_id = self.constraint_type_ids.add_subtype_id("sum")["id"]
        self.constraint_features.append(sum_subtype_id)
        sum_id = self.constraint_type_ids.add_node_id("sum")
        
        condition = constraint["condition"]
        condition_comparison_operator = condition["operator"]
        condition_operand = condition["operand"]
        comparison_operator_subtype_id = self.operator_type_ids.add_subtype_id(condition_comparison_operator)["id"]
        # comparison_operator_node_id = self.operator_type_ids.add_node_id(condition_comparison_operator)
        
        if condition_operand.isdigit(): # in this case, the sum is compared to a float
            new_comparison_node_id = self.operator_type_ids.add_node_id(subtype_name=condition_comparison_operator)
            self.operator_features.append(comparison_operator_subtype_id)    
            self.operator_to_constraint_edges.append([new_comparison_node_id, sum_id])
            self.operator_node_values[new_comparison_node_id] = float(condition_operand)
            
        else: # in this case, the sum is compared to a variable
            condition_operand_id = self.variable_type_ids.get_node_id(name=condition_operand)
            self.variable_to_constraint_edges.append([condition_operand_id, sum_id])
            new_comparison_node_id = self.operator_type_ids.add_node_id(subtype_name=condition_comparison_operator)
            self.operator_features.append(comparison_operator_subtype_id)    
            new_var_op_pair = [condition_operand_id, new_comparison_node_id]
            self.variable_to_operator_edges.append(new_var_op_pair)
            self.operator_to_constraint_edges.append([new_comparison_node_id, sum_id])
            self.operator_node_values[new_comparison_node_id] = 1.

        multiply_subtype_id = self.operator_type_ids.add_subtype_id("multiply")["id"]
        
        for i, variable in enumerate(variables):
            variable_id = self.variable_type_ids.get_node_id(name=variable)
            if coeffs[i] == 1:
                self.variable_to_constraint_edges.append([variable_id, sum_id])
            self.add_coeff_to_sum_constraint_subgraph(variable_id, coeffs[i], sum_id, multiply_subtype_id)
            
        self.constraint_to_objective_edges.append([sum_id, 0])
    
    def add_coeff_to_sum_constraint_subgraph(self, variable_id, coeff, constraint_id, multiply_subtype_id):
        new_multiply_node_id = self.operator_type_ids.add_node_id(subtype_name="multiply")
        self.operator_features.append(multiply_subtype_id)    
        new_var_op_pair = [variable_id, new_multiply_node_id]
        self.variable_to_operator_edges.append(new_var_op_pair)
        self.operator_to_constraint_edges.append([new_multiply_node_id, constraint_id])
        self.operator_node_values[new_multiply_node_id] = float(coeff)

    def add_intension_constraint_to_graph(self, constraint):
        # TODO: extend to other, non-graph coloring problems
        intension_subtype_id = self.constraint_type_ids.add_subtype_id("intension")["id"]
        intension_id = self.constraint_type_ids.add_node_id("intension")
        self.constraint_features.append(intension_subtype_id)
        for variable_name in constraint.children:
            variable_id = self.variable_type_ids.get_node_id(name=variable_name)
            self.variable_to_constraint_edges.append([variable_id, intension_id])
        
        self.constraint_to_objective_edges.append([intension_id, 0])

    def add_element_constraint_to_graph(self, constraint):
        index = constraint["index"]
        value = constraint["value"]
        element_list = constraint["list"]
        element_subtype_id = self.constraint_type_ids.add_subtype_id("element")["id"]
        element_id = self.constraint_type_ids.add_node_id("element")
        self.constraint_features.append(element_subtype_id)
        element_index_operator_subtype_id = self.operator_type_ids.add_subtype_id("element_index")["id"]
        element_index_operator_id = self.operator_type_ids.add_node_id("element_index")
        self.operator_features.append(element_index_operator_subtype_id)
        element_value_operator_subtype_id = self.operator_type_ids.add_subtype_id("element_value")["id"]
        element_value_operator_id = self.operator_type_ids.add_node_id("element_value")
        self.operator_features.append(element_value_operator_subtype_id)
        element_list_assignment_operator_subtype_id = self.operator_type_ids.add_subtype_id("element_list_assignment")["id"]

        for i, current_element in enumerate(element_list):
            element_assignment_name = "elm_assignment_" + str(current_element)
            element_assignment_id = self.operator_type_ids.get_node_id(name=element_assignment_name)
            if not element_assignment_id:
                element_assignment_id = self.operator_type_ids.add_node_id(subtype_name="element_list_assignment", name=element_assignment_name)
                self.operator_features.append(element_list_assignment_operator_subtype_id)
            if type(current_element) == float or type(current_element) == int:
                current_element_id = self.value_type_ids.get_node_id(name=current_element)
                self.value_to_operator_edges.append([current_element_id, element_assignment_id])
            else:
                current_element_id = self.variable_type_ids.get_node_id(name=current_element)
                self.variable_to_operator_edges.append([current_element_id, element_assignment_id])
            self.operator_to_constraint_edges.append([element_assignment_id, element_id])
        
        if type(index) == float or type(index) == int:
            index_id = self.value_type_ids.get_node_id(name=index)
            self.value_to_operator_edges.append([index_id, element_index_operator_id])
        else:
            index_id = self.variable_type_ids.get_node_id(name=index)
            self.variable_to_operator_edges.append([index_id, element_index_operator_id])
        self.operator_to_constraint_edges.append([element_index_operator_id, element_id])
        
        if type(value) == float or type(value) == int:
            value_id = self.value_type_ids.get_node_id(name=value)
            self.value_to_operator_edges.append([value_id, element_value_operator_id])
        else:
            value_id = self.variable_type_ids.get_node_id(name=value)
            self.variable_to_operato_edges.append([value_id, element_value_operator_id])
        self.operator_to_constraint_edges.append([element_value_operator_id, element_id])
        
        self.constraint_to_objective_edges.append([element_id, 0])

    def one_hot_encode(self, subtype_ids, id_to_value={}):
        """
        Creates a one-hot encoding of a list of integers.
        """
        # Get the maximum value in the list to determine the length of the one-hot vectors
        max_value = max(subtype_ids)
        # Create a one-hot encoded list for each integer
        one_hot_encoded = []
        for i, subtype_id in enumerate(subtype_ids):
            new_row = []
            for j in range(max_value + 1):
                if j == subtype_id:
                    new_row.append(id_to_value.get(i, 1))
                else:
                    new_row.append(0)
            one_hot_encoded.append(new_row)

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

    # filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\decision_tsp\text.xml"
    # filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\sample_problems\ClockTriplet-03-12_c22.xml"
    # filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\knapsack_instances\instance_1.xml"
    # filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\graph_coloring_instances\data0_0.xml"
    filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\decision_tsp\data\raw_elem\0.xml"
    instance = parse_instance(filepath)
    graph_builder = XCSP3GraphBuilder(instance, filepath)
    graph_builder.get_graph_representation()
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
    
    def print_variable_to_objective_edges(builder):
        for edge in builder.variable_to_objective_edges:
            formatted_edge = [builder.variable_type_ids.get_node_name(edge[0]), edge[1]]
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
    
    def print_constraint_to_objective_edges(builder):
        for edge in builder.constraint_to_objective_edges:
            formatted_edge = [builder.constraint_type_ids.get_node_name(edge[0]), edge[1]]
            print(formatted_edge)

    def print_operator_to_objective_edges(builder):
        for edge in builder.operator_to_objective_edges:
            formatted_edge = [builder.operator_type_ids.get_node_name(edge[0]), edge[1]]
            print(formatted_edge)

    print("\nvariable_to_value_edges\n")
    print_variable_to_value_edges(graph_builder)

    print("\nvariable_to_operator_edges\n")
    print_variable_to_operator_edges(graph_builder)

    print("\nvariable_to_constraint_edges\n")
    print_variable_to_constraint_edges(graph_builder)

    print("\nvariable_to_objective_edges\n")
    print_variable_to_objective_edges(graph_builder)

    print("\nvalue_to_operator_edges\n")
    print_value_to_operator_edges(graph_builder)

    print("\noperator_to_operator_edges\n")
    print_operator_to_operator_edges(graph_builder)

    print("\noperator_to_constraint_edges\n")
    print_operator_to_constraint_edges(graph_builder)

    print("\noperator_to_objective_edges\n")
    print_operator_to_objective_edges(graph_builder)

    print("\nconstraint_to_objective_edges\n")
    print_constraint_to_objective_edges(graph_builder)


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

    def print_objective_features(builder):
        for feature in builder.objective_features:
            print(f"Objective feature: {feature}")
    
    print("\nconstraint_features\n")
    print_constraint_features(graph_builder)
    print("\noperator_features\n")
    print_operator_features(graph_builder)
    print("\nvariable_features\n")
    print_variable_features(graph_builder)
    print("\nvalue_features\n")
    print_value_features(graph_builder)
    print("\nmeta_features\n")
    print_objective_features(graph_builder)
    
    a=1