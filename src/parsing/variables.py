import ast
import numpy as np
from typing import List


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


def get_array_dimensions(size):
    dimension_sizes = ast.literal_eval(
        "[" + size.replace("]", ",").replace("[", "") + "]")
    return dimension_sizes


def parse_variable_domain(raw_domain):
    domain = []
    sub_domains = [dom_element for dom_element in raw_domain.replace(
        "..", ",").split(" ")]
    for sub_domain in sub_domains:
        if not sub_domain:
            continue
        if "," in sub_domain:
            sub_domain = sub_domain.split(",")
            domain.append([int(sub_domain[0]), int(sub_domain[1])])
        else:
            domain.append([int(sub_domain)])
    return domain


def build_variable(variable_name, current_position, domain):
    full_variable_name = variable_name[0] + str(current_position)
    new_var = Variable(full_variable_name, domain)

    return new_var


def parse_variable(var_name, dim_index, dimensions, current_position, domain, variables, sizes, allocated_variables):
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

def parse_array_variables(array_vars):
    instance_variables = []
    for array in array_vars:
        array_dimensions = get_array_dimensions(array.attrib["size"])
        array_domains_parsed = np.zeros(array_dimensions)
        for variable in array:
            if variable.tag == "domain":
                domain = parse_variable_domain(variable.text)
                var_names = variable.attrib["for"].split()
                for new_var_name in var_names:
                    current_array_dims = new_var_name.replace(
                        "]", "").split('[')[1:]
                    parse_variable(new_var_name, 0, current_array_dims, [
                    ], domain, instance_variables, array_dimensions, array_domains_parsed)
    
    return instance_variables

# TODO: fill in "other" variables
# TODO: parse non-array variables