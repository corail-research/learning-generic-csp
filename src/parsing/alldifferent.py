import ast
from typing import List, Dict
import xml

from variables import *


def parse_alldiff_constraint(constraint: xml.etree.ElementTree.Element, instance_variables: Dict) -> List[Dict]:
    """Parses one allDifferent constraint in a given problem

    Args:
        raw_alldiff_constraint (xml.etree.ElementTree.Element): A list of XML elements representing the raw allDifferent constraints in the problem
        instance_variables (Dict): variables involved in the problem
    Returns:
        List[Dict]: A list of dictionaries representing the parsed allDifferent constraints
    """
    # for constraint in raw_alldiff_constraints:
    # If the constraint contains a "list" element, parse it as a list constraint
    if constraint.find("list") is not None:
        parsed_constraint = parse_list_and_exceptions(constraint, instance_variables)
    # Otherwise, parse it as a base allDifferent constraint
    else:
        parsed_constraint = parse_base_alldiff_constraint(constraint, instance_variables)

    return "allDifferent", parsed_constraint


def parse_list_and_exceptions(root: xml.etree.ElementTree.Element, instance_variables: Dict) -> Dict:
    """Parses generic allDifferent constraints; the ones like:
    <allDifferent id = "c2" >
        <list > v1 v2 v3 v4 < /list >
        <list > w1 w2 w3 w4 < /list >
        <list > z1 z2 z3 z4 < /list >
    </allDifferent>

    Args:
        root (xml.etree.ElementTree.Element): generic allDifferent constraint

    Returns:
        constraint (Dict): example: {
            "lists" : [
                (v1, v2, v3, v4)
                (w1, w2, w3, w4)
                (z1, z2, z3, z4)
            ]
        }
    """
    lists = []
    for elem in root.findall("list"):
        new_vars = parse_arg_variables(elem.text, instance_variables)
        lists.append(new_vars)

    return lists
        

def parse_base_alldiff_constraint(root: xml.etree.ElementTree.Element, instance_variables: Dict) -> Dict:
    """Parses basic allDifferent constraints; the ones like:
    <allDifferent>
       x1 x2 x3 x4 x5
     </allDifferent>

    Args:
        root (xml.etree.ElementTree.Element): allDifferent constraint
        instance_variables (Dict): variables involved in the problem

    Returns:
        constraint (List[str]): example: ["x1", "x2", "x3", "x4", "x5"]
        Note: "exceptions" key can be None or list
    """
    constraint = parse_arg_variables(root.text.strip(), instance_variables)

    return constraint

def parse_alldiff_group(group: xml.etree.ElementTree.Element, instance_variables: Dict) -> Dict:
    """
    Parse a group of allDifferent constraints in a given problem.

    """
    parsed_alldiff_constraint = []
    args = group.findall("args")
    for raw_arg in args:
        arg_format = raw_arg.text.strip()
        new_constraint = parse_arg_variables(arg_format, instance_variables)
        parsed_alldiff_constraint.append(new_constraint)
    
    return "allDifferent", parsed_alldiff_constraint