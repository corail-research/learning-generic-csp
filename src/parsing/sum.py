import ast
from typing import Dict
import xml.etree.ElementTree as ET

class SumNode:
    def __init__(self, variables, operator, operand):
        self.variables = variables
        self.condition = {'operator': operator, 'operand': operand}


def parse_sum(sum_element:ET.Element, instance_variables:Dict):
    """Parse a <sum> element into a dictionary.

    Args:
        sum_element: An ElementTree element representing a <sum> element.
        instance_variables: variables involved in the problem
    Returns:
        A dictionary containing the parsed <sum> element.
    """
    # Get the <list>, <coeffs>, and <condition> sub-elements
    list_element = sum_element.find("list")
    coeffs_element = sum_element.find("coeffs")
    condition_element = sum_element.find("condition")
    
    # Parse the variables from the <list> element
    list_text = list_element.text.strip()
    if "[]" in list_text:
        array_name = list_text[:list_text.find("[")]
        variable_name = list_text[:list_text.find("[]")]
        implicit_dimension = instance_variables.array_variables[array_name].get_all_variables_from_implicit_subarray_name(list_text)
        variables = [f"{variable_name}[{i}]" for i in range(len(implicit_dimension))]
    else:
        variables = list_text.split()

    # Parse the coefficients from the <coeffs> element
    coeffs_raw = coeffs_element.text.strip().split()
    coeffs = []
    for c in coeffs_raw:
        if "x" in c:
            value, occurence = c.split("x")
            for _ in range(int(occurence)):
                coeffs.append(int(value))
        else:
            coeffs.append(int(c))
    if coeffs_raw is None:
        coeffs = [1 for _ in range(len(variables))]

    # Parse the condition from the <condition> element
    condition_raw = condition_element.text.strip().replace("(", "").replace(")", "").split(",")
    condition = {"operator": condition_raw[0], "operand": condition_raw[1].strip()}

    # Construct the parsed <sum> element as a dictionary
    parsed_sum = {"variables": variables, "coeffs": coeffs, "condition": condition}

    return "sum", parsed_sum

def parse_sum_group(group:ET.Element, variables):
    pass


if __name__ == "__main__":
    # def test_parse_expression():
    expression = """<sum>
    <list> x1 x2 x3 </list>
    <coeffs> 1 2 3 </coeffs>
    <condition> (gt, y) </condition>
    </sum>
    """
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)
    # assert operator == "gt"
    # assert coeffs == [1, 2, 3]
    # assert children == ["x1", "x2", "x3", "y"]

    expression = """<sum>
    <list> y1 y2 y3 y4 </list>
    <coeffs> 4 2 3 1 </coeffs>
    <condition> (eq, 10) </condition>
    </sum>"""
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)
    # assert operator == "eq"
    # assert coeffs == [4, 2, 3, 1]
    # assert children == ["y1", "y2", "y3", "y4"]

    expression = """<sum>
    <list> w[] </list>
    <coeffs> 1x4 2x2 </coeffs>
    <condition> (le, 10) </condition>
    </sum>"""
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)
    # assert operator == "le"
    # assert coeffs == [1,1,1,1, 2, 2]
    # assert children == ["w[0]", "w[1]", "w[2]", "w[3]"]

    expression = """<sum>
    <list> w[] </list>
    <coeffs> 1 1 1 1 2 2 </coeffs>
    <condition> (le, 10) </condition>
    </sum>"""
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)
    # assert operator == "le"
    # assert coeffs == [1, 1, 1, 1, 2, 2]
    # assert children == ["w[0]", "w[1]", "w[2]", "w[3]", "w[4]", "w[5]"]