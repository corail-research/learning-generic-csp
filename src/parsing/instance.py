import variables as variables_parsing
from constraints import parse_constraint_section
from typing import List
import xml.etree.ElementTree as ET


class XCSP3Instance:
    def __init__(self, variables: variables_parsing.InstanceVariables, constraints: List):
        self.variables = variables
        self.constraints = constraints
    
    def get_all_variables(self):
        variable_counter = 0
        all_vars = {}
        variable_mapping = {} #mapping from variable_name to id -for graph generation
        for _, array_var in self.variables.array_variables.items():
            for variable_name, variable in array_var.variables.items():
                all_vars[variable_name] = variable
                variable_mapping[variable_name] = variable_counter
                variable_counter += 1
        
        for _, variable in self.variables.integer_variables.items():
            all_vars[variable.name] = variable
            variable_mapping[variable.name] = variable_counter
            variable_counter += 1

        return all_vars, variable_mapping

def parse_instance(filepath:str) -> XCSP3Instance:
    """Parses an XCSP3 instance file."""
    root = ET.parse(filepath)
    variables = root.findall("variables")
    instance_variables = variables_parsing.parse_all_variables(variables)
    constraints = root.findall("constraints")[0]
    constraints = parse_constraint_section(instance_variables, constraints)

    return XCSP3Instance(instance_variables, constraints)

if __name__ == "__main__":
    import os
# test files are all xml files in C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCSP23
    test_files_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCSP23"
    files = [os.path.join(test_files_path, file) for file in os.listdir(test_files_path)]

    for filepath in files:
        if "lzma" in filepath:
            continue
        instance = parse_instance(filepath)
        a=1