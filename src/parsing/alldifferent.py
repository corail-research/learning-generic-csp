import ast
from typing import List, Dict
import xml


def parse_alldiff_constraint(constraint: xml.etree.ElementTree.Element) -> List[Dict]:
    """Parses all allDifferent constraints in a given problem

    Args:
        raw_alldiff_constraints (xml.etree.ElementTree.Element): A list of XML elements representing the raw allDifferent constraints in the problem

    Returns:
        List[Dict]: A list of dictionaries representing the parsed allDifferent constraints
    """
    # for constraint in raw_alldiff_constraints:
    # If the constraint contains a "list" element, parse it as a list constraint
    if constraint.find("list") is not None:
        parsed_constraint = parse_list_and_exceptions(constraint)
        # alldiff_constraints.append(parsed_constraint)
    # If the constraint contains a "matrix" element, parse it as a matrix constraint
    elif constraint.find("matrix") is not None:
        parsed_constraint = parse_matrix(constraint)
        # alldiff_constraints.append(parsed_constraint)
    # Otherwise, parse it as a base allDifferent constraint
    else:
        parsed_constraint = parse_base_alldiff_constraint(constraint)
        # alldiff_constraints.append(parsed_constraint)

    return "allDifferent", parsed_constraint


def parse_list_and_exceptions(root: xml.etree.ElementTree.Element):
    """Parses generic allDifferent constraints; the ones like:
    <allDifferent id = "c2" >
        <list > v1 v2 v3 v4 < /list >
        <list > w1 w2 w3 w4 < /list >
        <list > z1 z2 z3 z4 < /list >
        <except > (0, 0, 0, 0) < /except >
    </allDifferent>

    Args:
        root (xml.etree.ElementTree.Element): generic allDifferent constraint

    Returns:
        constraint (Dict): example: {
            "lists" : [
                (v1, v2, v3, v4)
                (w1, w2, w3, w4)
                (z1, z2, z3, z4)
            ],
            "exceptions": ((0,0,0,0))
        }
    """
    lists = []
    for elem in root.findall("list"):
        items = elem.text.replace("[]", "").strip().split()
        lists.append(items)

    except_elem = root.find("except")
    exceptions = None
    if except_elem is not None:
        exceptions = ast.literal_eval(except_elem.text.strip())
    if exceptions is not None and not isinstance(exceptions, tuple):
        exceptions = (exceptions)
    result = {"lists": lists, "exceptions": exceptions}

    return result


def parse_matrix(root: xml.etree.ElementTree.Element) -> Dict:
    """Parses matrix allDifferent constraints; the ones like:
    < allDifferent >
        <matrix >
            (x1, x2, x3, x4, x5)
            (y1, y2, y3, y4, y5)
            (z1, z2, z3, z4, z5)
        </matrix >
    </allDifferent >

    Args:
        root (xml.etree.ElementTree.Element): matrix allDifferent constraint

    Returns:
        constraint (Dict): example: {
            "matrix" : [
                (x1, x2, x3, x4, x5)
                (y1, y2, y3, y4, y5)
                (z1, z2, z3, z4, z5)
            ]
        }
    """
    matrix = {}
    for matrix_element in root.findall("matrix"):
        matrix["matrix"] = [tuple(row.strip().strip("()").split(
            ",")) for row in matrix_element.text.strip().split("\n")]
    return matrix
        

def parse_base_alldiff_constraint(root: xml.etree.ElementTree.Element) -> Dict:
    """Parses basic allDifferent constraints; the ones like:
    <allDifferent>
       x1 x2 x3 x4 x5
     </allDifferent>

    Args:
        root (xml.etree.ElementTree.Element): _description_

    Returns:
        constraint (Dict): example: {
            "lists" : [[x, y, z]],
            "exceptions": [0,0,0]

        }
        Note: "exceptions" key can be None or list
    """
    constraint = {}
    constraint["lists"] = root.text.strip().split()
    constraint["exceptions"] = None

    return constraint