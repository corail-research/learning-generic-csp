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
        variables = instance_variables.array_variables[array_name].get_all_variables_from_implicit_subarray_name(list_text)
    if ".." in list_text:
        array_name = list_text[:list_text.find("[")]
        variables = instance_variables.array_variables[array_name].get_all_variables_from_shortened_subarray_name(list_text)
    else:
        variables = list_text.split()

    # Parse the coefficients from the <coeffs> element
    if coeffs_element is None:
        coeffs = [1 for _ in range(len(variables))]
    else:
        coeffs = []
        coeffs_raw = coeffs_element.text.strip().split()
        for c in coeffs_raw:
            if "x" in c:
                value, occurence = c.split("x")
                for _ in range(int(occurence)):
                    coeffs.append(int(value))
            else:
                coeffs.append(int(c))

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

    expression = """<sum>
    <list> y1 y2 y3 y4 </list>
    <coeffs> 4 2 3 1 </coeffs>
    <condition> (eq, 10) </condition>
    </sum>"""
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)

    expression = """<sum>
    <list> w[] </list>
    <coeffs> 1x4 2x2 </coeffs>
    <condition> (le, 10) </condition>
    </sum>"""
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)

    expression = """<sum>
    <list> w[] </list>
    <coeffs> 1 1 1 1 2 2 </coeffs>
    <condition> (le, 10) </condition>
    </sum>"""
    expression = ET.fromstring(expression)
    constraint_type, parsed_constraint = parse_sum(expression)