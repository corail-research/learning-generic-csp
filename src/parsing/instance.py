import variables as variables_parsing
from constraints import parse_constraint_section
from typing import List
import xml.etree.ElementTree as ET

class Objective:
    def __init__(self, objective_type, variables, coeffs, optimal_value):
        self.objective_type = objective_type
        self.variables = variables
        self.coeffs = coeffs
        self.optimal_value = optimal_value

class XCSP3Instance:
    def __init__(self,
                    variables: variables_parsing.InstanceVariables, 
                    constraints: List,
                    objective: Objective=None,
                    optimal_deviation_factor: float=None,
                    optimal_deviation_difference: float=None,
                    label: str=None
                 ):
        self.variables = variables
        self.constraints = constraints
        self.objective = objective
        self.optimal_value = objective.optimal_value
        self.optimal_deviation_factor = optimal_deviation_factor
        self.optimal_deviation_difference = optimal_deviation_difference
        self.label = label
    
    def get_all_variables(self):
        all_vars = {}
        for _, array_var in self.variables.array_variables.items():
            for variable_name, variable in array_var.variables.items():
                all_vars[variable_name] = variable
        
        for _, variable in self.variables.integer_variables.items():
            all_vars[variable.name] = variable

        return all_vars

def parse_instance(filepath:str, optimal_deviation_factor: float=None, optimal_deviation_difference: float=None, label: str=None) -> XCSP3Instance:
    """Parses an XCSP3 instance file."""
    root = ET.parse(filepath)
    variables = root.findall("variables")
    instance_variables = variables_parsing.parse_all_variables(variables)
    constraints = root.findall("constraints")[0]
    constraints = parse_constraint_section(instance_variables, constraints)
    objective_element = root.findall("objectives")
    if objective_element:
        objective = parse_objective(objective_element, instance_variables)

    return XCSP3Instance(instance_variables, constraints, objective, optimal_deviation_factor, optimal_deviation_difference, label)

def parse_objective(objective_element: ET.Element, instance_variables: variables_parsing.InstanceVariables):
    """Parses an objective element in a given problem

    Args:
        objective_element (ET.Element): The objective element to parse
        instance_variables (variables_parsing.InstanceVariables): The variables in the instance

    Returns:
        Tuple[str, Dict]: A tuple containing the objective type and the parsed objective
    """
    if objective_element is not None:
        optimal_element = objective_element[0].findall("optimal")
        if optimal_element:
            optimal_value = float(optimal_element[0].text)
        else:
            optimal_value = None
        minimize_element = objective_element[0].findall("minimize")
        maximize_element = objective_element[0].findall("maximize")
        variables_in_objective = []
        if minimize_element:
            objective_type, variables_in_objective, coeffs = parse_objective_type(minimize_element, instance_variables)
        else:
            objective_type, variables_in_objective, coeffs = parse_objective_type(maximize_element, instance_variables)
        
        return Objective(objective_type, variables_in_objective, coeffs, optimal_value)
    else:
        return None

def parse_objective_type(element, instance_variables):
    """Parses an objective minimize/maximize element in a given problem"""
    variables_in_objective = []
    objective_type = element[0].attrib["type"]
    element_text = element[0].text.strip()
    if element_text:
        raw_variables = element_text.split()
    else:
        raw_variables = element[0].findall("list")[0].text.split()
    for variable_name in raw_variables:
        if "[" not in variable_name:
            variables_in_objective.append(variable_name)
        else:
            variable_array_name = variable_name.split("[")[0]
            new_variables = instance_variables.array_variables[variable_array_name].get_all_variables_from_shortened_subarray_name(variable_name)
            variables_in_objective.extend(new_variables)
    coeffs_element = element[0].findall("coeffs")
    if coeffs_element:
        coeffs = [ int(i) for i in coeffs_element[0].text.strip().split()]
    else:
        coeffs = None
    
    return objective_type, variables_in_objective, coeffs


if __name__ == "__main__":
    import os
# test files are all xml files in C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCSP23
    test_files_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCOP23"
    files = [os.path.join(test_files_path, file) for file in os.listdir(test_files_path)]
    filepath = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCOP23\Benzenoide-07_mini_c23.xml"
    # filepath = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCOP\Triangular-10_mc22.xml..xml"
    # for filepath in files:
    #     if "lzma" in filepath:
    #         continue
    instance = parse_instance(filepath)
        # a=1