import ast
import numpy as np
from typing import List
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


def get_array_dimensions(size):
    dimension_sizes = ast.literal_eval(
        "[" + size.replace("]", ",").replace("[", "") + "]")
    return dimension_sizes


def parse_variable_domain(raw_domain):
    domain = []
    sub_domains = [dom_element for dom_element in raw_domain.replace("..", ",").replace("infinity", "inf").split(" ")]
    for sub_domain in sub_domains:
        if not sub_domain:
            continue
        if "," in sub_domain:
            start, end = sub_domain.split(",")
            lower_bound = int(start) if "inf" not in start else float(start)
            upper_bound = int(end) if "inf" not in end else float(end)
            domain.append([lower_bound, upper_bound])
        else:
            domain.append([int(sub_domain)])
    return domain


def build_variable(variable_name, current_position, domain):
    position_as_name = str(current_position).replace(", ", "][")
    full_variable_name = variable_name + position_as_name
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
    instance_variables = {}
    other_domain = None  # domain for other variables
    for array in array_vars:
        array_variables = []
        array_name = array.attrib["id"]
        array_dimensions = get_array_dimensions(array.attrib["size"])
        array_domains_parsed = np.zeros(array_dimensions)
        for variable in array:
            if variable.tag == "domain":
                domain = parse_variable_domain(variable.text)
                var_names = variable.attrib["for"].split()
                if var_names == "others":
                    other_domain = domain
                    continue
                for new_var_name in var_names:
                    current_array_dims = new_var_name.replace("]", "").split('[')[1:]
                    parse_variable(array_name, 0, current_array_dims, [
                    ], domain, array_variables, array_dimensions, array_domains_parsed)

        for i, val in enumerate(array_domains_parsed):
            if val == 0:
                real_index = list(np.unravel_index(i, tuple(array_dimensions)))
                new_var = build_variable(array_name, real_index, other_domain)
                array_variables.append(new_var)
        instance_variables[array_name] = array_variables

    return instance_variables

def parse_integer_variables(int_vars):
    instance_variables = {}
    for var in int_vars:
        variable_name = var.attrib["id"]
        domain = parse_variable_domain(var.text)
        new_var = build_variable(variable_name, "", domain)
        instance_variables[variable_name] = new_var
    return instance_variables
        

if __name__ == "__main__":
    file_path = r"..\sample_problems\sample_problem_test\AircraftLanding-table-airland01_mc22.xml"
    root = ET.parse(file_path)
    variables = root.findall("variables")
    array_vars = variables[0].findall("array")
    integer_vars = variables[0].findall("var")
    parsed_array_variables = parse_array_variables(array_vars)