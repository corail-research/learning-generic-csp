import ast
from typing import Dict, List
import xml.etree.ElementTree as ET
from parsing.variable_parsing import *

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
    
    list_text = list_element.text.strip()
    variables = parse_arg_variables(list_text, instance_variables)
    coeffs = parse_coeffs(coeffs_element, len(variables))

    # Parse the condition from the <condition> element
    condition_raw = condition_element.text.strip().replace("(", "").replace(")", "").split(",")
    condition = {"operator": condition_raw[0], "operand": condition_raw[1].strip()}
    # Construct the parsed <sum> element as a dictionary
    parsed_sum = {"variables": variables, "coeffs": coeffs, "condition": condition}

    return "sum", parsed_sum

def parse_sum_group(group:ET.Element, instance_variables: Dict):
    """Parse a <sum> element into a dictionary.
    Args:
        sum_element: An ElementTree element representing a <sum> element.
        instance_variables: variables involved in the problem
    Returns:
        A dictionary containing the parsed <sum> element.
    """
    parsed_group = []
    # Parse the <sum> element
    sum_element = group.find("sum")
    condition_element = sum_element.find("condition")
    condition_raw = condition_element.text.strip().replace("(", "").replace(")", "").split(",")
    condition = {"operator": condition_raw[0], "operand": condition_raw[1].strip()}
    args_elements = group.findall("args")

    for args_element in args_elements:
        args_text = args_element.text.strip()
        variables = parse_arg_variables(args_text, instance_variables)
        coeffs_element = sum_element.find("coeffs")
        coeffs = parse_coeffs(coeffs_element, len(variables))
        parsed_sum = {"variables": variables, "coeffs": coeffs, "condition": condition}
        parsed_group.append(parsed_sum)

    return "sum", parsed_group

def parse_coeffs(coeffs_element: ET.Element, num_variables: int) -> List[int]:
    """Parse the coefficients from the <coeffs> element of a <sum> element.

    Args:
        coeffs_element: An ElementTree element representing a <coeffs> element.
        num_variables: number of variables in the sum

    Returns:
        coeffs: A list of coefficients for terms in a sum constraint.
    """
    # Parse the coefficients from the <coeffs> element
    if coeffs_element is None:
        coeffs = [1 for _ in range(num_variables)]
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
    
    return coeffs


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