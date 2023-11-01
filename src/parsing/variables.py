import ast
import numpy as np
from typing import Dict, List
import xml
import xml.etree.ElementTree as ET


class Variable:
    """
    Attributes:
    - name (str): variable's name
    - domain (List[List[int]]): all intervals comprising the variable's domain; ex: [[1,5], [8], [11, 16]] means that the variable's domain is {1, 2, 3, 4, 5, 8, 11, 12, 13, 14, 15, 16}
    """
    def __init__(self, name: str, domain: List[List[int]]):
        self.name = name
        self.domain = domain

    def __repr__(self):
        return repr(f"""Name: {self.name} Domain: {self.domain}""")

class VariableArray:
    def __init__(self, size:List[int], variables: List[Variable]):
        self.size = size
        self.variables = {var.name: var for var in variables}

    def get_all_variables_from_implicit_subarray_name(self, subarray_name):
        """Get the implicit dimensions of a subarray with implicit dimensions
        for example:
        - if array x has dimensions [14, 15] and we have subarray x[1][], we would get 15 variables:
            x[1][0], x[1][1], ..., x[1][14]
        - if array y has dimensions [14, 15, 16] and we have subarray y[][1][], we would get 14 * 16 variables:
            y[0][1][0], y[0][1][1], ..., y[0][1][15], y[1][1][0], y[1][1][1], ..., y[1][1][15], ..., y[13][1][0], y[13][1][1], ..., y[13][1][15]

        Args:
            variable (str): A variable name with the form "x[1][2][3]"

        Returns:
            A list of the implicit dimensions
        """
        array_name = subarray_name[:subarray_name.find("[")]
        variable_names = [array_name]
        raw_dimensions = ast.literal_eval(subarray_name[subarray_name.find("["):].replace("]", "],"))
        for i, dim in enumerate(raw_dimensions):
            if not dim:
                implicit_size = self.size[i]
                new_variable_names = []
                for variable_name in variable_names:
                    for j in range(implicit_size):
                        new_variable_names.append(f"{variable_name}[{j}]")
                variable_names = new_variable_names
            else:
                for i in range(len(variable_names)):
                    variable_names[i] += f"{dim}"

        return variable_names
    
    def get_all_variables_from_shortened_subarray_name(self, subarray_name):
        """Get the dimensions of a subarray with shortnened dimensions
        for example:
        - if array x has dimensions [14, 15] and we have subarray x[1][0..3], we would get 4 variables:
            x[1][0], x[1][1], x[1][2], x[1][3]
        - if array y has dimensions [2, 15, 3] and we have subarray y[][1][], we would get 2 * 3 variables:
            y[0][1][0], y[0][1][1], y[0][1][2],
            y[1][1][0], y[1][1][1], y[1][1][2]

        Args:
            variable (str): A variable name with the form "x[1][2][1..3]"

        Returns:
            A list of the implicit dimensions
        """
        subarray_name = subarray_name.replace("..", ",")
        array_name = subarray_name[:subarray_name.find("[")]
        variable_names = [array_name]
        raw_dimensions = ast.literal_eval(subarray_name[subarray_name.find("["):].replace("]", "],"))
        for i, dim in enumerate(raw_dimensions):
            if not dim:
                implicit_size = self.size[i]
                new_variable_names = []
                for variable_name in variable_names:
                    for j in range(implicit_size):
                        new_variable_names.append(f"{variable_name}[{j}]")
                variable_names = new_variable_names
            elif len(dim) > 1:
                new_variable_names = []
                for variable_name in variable_names:
                    for j in range(dim[0], dim[1] + 1):
                        new_variable_names.append(f"{variable_name}[{j}]")
                variable_names = new_variable_names
            else:
                for i in range(len(variable_names)):
                    variable_names[i] += f"{dim}"

        return variable_names

class InstanceVariables:
    def __init__(self, integer_variables: Dict[str, Variable], array_variables: Dict[str, VariableArray]):
        self.integer_variables = integer_variables
        self.array_variables = array_variables
        self.domain = self.get_domain()
    
    def contains_variable(self, variable_name:str):
        if "[]" in variable_name:
            array_name = variable_name[:variable_name.find("[")]
            return array_name in self.array_variables
        if variable_name in self.integer_variables:
            return True
        for _, array_vars in self.array_variables.items():
            for var_name, _ in array_vars.variables.items():
                if var_name == variable_name:
                    return True
        return False
    
    def get_domain(self):
        """Get the union of domain of all variables in the instance"""
        domain = set()
        for _, var in self.integer_variables.items():
            domain.update(var.domain)
        for _, array_vars in self.array_variables.items():
            for _, var in array_vars.variables.items():
                domain.update(var.domain)
        return sorted(list(domain))

def parse_all_variables(variables:List[xml.etree.ElementTree.Element]):
    """Parses all variables in a given instance

    Args:
        variables (List[xml.etree.ElementTree.Element]): root of variables in given XCSP3 problem
    """
    array_vars = variables[0].findall("array")
    integer_vars = variables[0].findall("var")
    parsed_integer_variables = parse_integer_variables(integer_vars)
    parsed_array_variables = parse_array_variables(array_vars)
    
    return InstanceVariables(parsed_integer_variables, parsed_array_variables)

def parse_array_variables(array_vars: List[xml.etree.ElementTree.Element]) -> dict:
    """Parse all array variables and return the output as a dict where keys are the name of the arrays and values are of type List[Variable]
    Args:
        array_vars: list containing all array variables

    Returns:
        example: {
            "x": [Variable("x[0]", [1, 4]), Variable("x[1]", [1, 5])],
            "y": [Variable("y[0]", [4, 7]), Variable("y[1]", [1, 9])]
        }
    """
    instance_variables = {}
    for array in array_vars:
        # other_domain = None  # domain for other variables
        domain = None
        array_variables = []
        array_name = array.attrib["id"]
        array_dimensions = get_array_dimensions(array.attrib["size"])
        array_domains_parsed = np.zeros(array_dimensions)
        for variable in array:
            if variable.tag == "domain":
                domain = parse_variable_domain(variable.text)
                var_names = variable.attrib["for"].split()
                for new_var_name in var_names:
                    current_array_dims = new_var_name.replace("]", "").split('[')[1:]
                    parse_variable(
                        array_name,
                        0,
                        current_array_dims,
                        [],
                        domain,
                        array_variables,
                        array_dimensions,
                        array_domains_parsed
                    )
        if domain is None:
            domain = parse_variable_domain(array.text)

        for i, val in enumerate(array_domains_parsed.flatten()):
            if val == 0:
                real_index = list(np.unravel_index(i, tuple(array_dimensions)))
                new_var = build_variable(array_name, real_index, domain)
                array_variables.append(new_var)
        new_array = VariableArray(array_dimensions, array_variables)
        instance_variables[array_name] = new_array

    return instance_variables

def parse_variable(
    var_name:str, 
    dim_index:0, 
    dimensions:List[str],
    current_position:List[int],
    domain:List[List[int]],
    variables:List[Variable],
    sizes:List[int],
    allocated_variables:np.array):
    """Recursively parse variables

    Args:
        var_name: name of array variable 
        dim_index : index of the current position in the array
        dimensions : dimension where the variable domains are defined
        current_position : suffix to append to the variable name
        domain : domain for the variable(s); defined as intervals
        variables : contains variables created
        sizes : dimensions of the variable array
        allocated_variables : np.array of binary values; keeps track of which values were allocated
    """
    if dim_index == len(dimensions):
        new_variable = build_variable(var_name, current_position, domain)
        variables.append(new_variable)
        allocated_variables[tuple(current_position)] = 1
        return
    current_dim = dimensions[dim_index]
    if current_dim:
        if ".." in current_dim:
            dims_range = [int(dim) for dim in current_dim.split("..")]
            lower_bound = dims_range[0]
            upper_bound = dims_range[1] + 1
        else:
            lower_bound = int(current_dim)
            upper_bound = int(current_dim) + 1
        for dim in range(lower_bound, upper_bound):
            current_position.append(dim)
            parse_variable(var_name, dim_index+1, dimensions, current_position, domain, variables, sizes, allocated_variables)
            current_position.pop()
    else:
        for dim in range(sizes[dim_index]):
            current_position.append(dim)
            parse_variable(var_name, dim_index+1, dimensions, current_position, domain, variables, sizes, allocated_variables)
            current_position.pop()

def build_variable(variable_name:str, current_position:List[int], domain:List[List[int]]):
    """Create a variable
    """
    position_as_name = str(current_position).replace(", ", "][")
    full_variable_name = variable_name + position_as_name
    new_var = Variable(full_variable_name, domain)

    return new_var

def parse_integer_variables(int_vars):
    instance_variables = {}
    for var in int_vars:
        variable_name = var.attrib["id"]
        base_var = var.attrib.get("as")
        if var.text is not None:
            domain = parse_variable_domain(var.text)
        elif base_var is not None:
            domain = instance_variables[base_var].domain
        else:
            raise("Unrecognized variable domain")
        new_var = build_variable(variable_name, "", domain)
        instance_variables[variable_name] = new_var
    return instance_variables

def get_array_dimensions(size:str) -> int:
    """Compute array dimensions
    """
    dimension_sizes = ast.literal_eval(
        "[" + size.replace("]", ",").replace("[", "") + "]")
    return dimension_sizes

def parse_variable_domain(raw_domain:str):
    """Get the domain 

    Args:
        raw_domain : domain expressed as string

    Returns:
        domain: List[List[int]]
    """
    domain = []
    sub_domains = [dom_element for dom_element in raw_domain.replace("..", ",").replace("infinity", "inf").split(" ")]
    for sub_domain in sub_domains:
        if not sub_domain:
            continue
        if "," in sub_domain:
            start, end = sub_domain.split(",")
            lower_bound = int(start) if "inf" not in start else float(start)
            upper_bound = int(end) if "inf" not in end else float(end)
            domain_values = list(range(lower_bound, upper_bound + 1))
            domain.extend(domain_values)
        else:
            domain.append(int(sub_domain))
    return domain
        
def parse_arg_variables(arg: str, instance_variables: Dict) -> List[str]:
    """Parses the variables involved in an <args> section

    Args:
        arg (str): <arg> element in sum group or <list> in basic sum constraint
        instance_variables (Dict): variables involved in the problem

    Returns:
        List[str]: _description_
    """
    
    variables = []
    arrays = arg.split()
    for array in arrays:
        array_name = array[:array.find("[")]
        # if "[]" in array:
        #     new_vars = instance_variables.array_variables[array_name].get_all_variables_from_implicit_subarray_name(array)
        #     variables.extend(new_vars)
        if ".." in array or "[]" in array:
            new_vars = instance_variables.array_variables[array_name].get_all_variables_from_shortened_subarray_name(array)
            variables.extend(new_vars)
        else:
            variables.extend(array.split())
    
    return variables

if __name__ == "__main__":
    import os
    directory = r"C:/Users/leobo/Desktop/École/Poly/SeaPearl/instancesXCSP22/MiniCOP"
    all_files = [os.path.join(directory, file) for file in os.listdir(directory)]
    directory = r"C:/Users/leobo/Desktop/École/Poly/SeaPearl/instancesXCSP22/MiniCSP"
    all_files += [os.path.join(directory, file) for file in os.listdir(directory)]
    for file_path in all_files:
        root = ET.parse(file_path)
        variables = root.findall("variables")
        # array_vars = variables[0].findall("array")
        # integer_vars = variables[0].findall("var")
        # parsed_integer_variables = parse_integer_variables(integer_vars)
        # parsed_array_variables = parse_array_variables(array_vars)
        parsed_variables = parse_all_variables(variables)

        a = 1