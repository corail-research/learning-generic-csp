import ast
from typing import List, Dict
import xml



def parse_all_different_constraints(raw_alldiff_constraints: List[xml.etree.ElementTree.Element]) -> List[Dict]:
    """Parses all allDifferent constraints in a given problem

    Args:
        raw_alldiff_constraints (xml.etree.ElementTree.Element): _description_

    Returns:
        _type_: _description_
    """
    alldiff_constraints = []
    for constraint in raw_alldiff_constraints:
        if constraint.find("list") is not None:
            parsed_constraint = parse_list_and_exceptions(constraint)
            alldiff_constraints.append(parsed_constraint)
        elif constraint.find("matrix") is not None:
            parsed_constraint = parse_matrix(constraint)
            alldiff_constraints.append(parsed_constraint)
        else:
            parsed_constraint = parse_base_alldiff_constraint(constraint)
            alldiff_constraints.append(parsed_constraint)

    return alldiff_constraints


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
    

def parse_block(block: xml.etree.ElementTree.Element) -> List[Dict]:
    """Parses a constraint block. Works only for extension constraints. Returns a list of dicts from the parse_extension_constraint
    function

    Args:
        block: element of name "block". Contains constraints grouped together

    Returns:
        constraints (List[Dict]): parsed extension constraints
    """
    constraints = []
    for child in block:
        name = child.tag
        if name == "extension":
            if child.find("supports") is not None:
                new_constraint = parse_extension_constraint(child)
            else:
                new_constraint = parse_negative_extension_constraint(child)
        constraints.append(new_constraint)

    return constraints


def parse_extension_constraint(constraint: xml.etree.ElementTree.Element) -> Dict:
    """Parse an individual extension constraint defined with supports; that is values the variables CAN take. 
    Extension constraints defined with exclusion are parsed with function parse_negative_extension_constraint
    Returns a dict like:
    {"variables": [x, y], "tuples": ((1, 2), (2, 3))}

    Args:
        constraint (xml.etree.ElementTree.Element): element of "extension" ; extension constraint

    Returns:
        parsed_constraint: dict containing the parsed constraint
    """
    parsed_constraint = {}
    variables = constraint.find("list").text.strip().split()
    parsed_constraint["variables"] = variables
    tuples = ast.literal_eval(constraint.find(
        "supports").text.strip().replace(")", "),"))
    parsed_constraint["tuples"] = tuples

    return parsed_constraint


def parse_negative_extension_constraint(constraint: xml.etree.ElementTree.Element) -> Dict:
    """Parse an individual extension constraint defined with conflicts; that is values the variables can't take.
    Returns a dict like:
    {"variables": [x, y], "tuples": ((1, 2), (2, 3))}, which means variables (x, y) can't take values (1, 2) or (2, 3)

    Args:
        constraint (xml.etree.ElementTree.Element): element of "extension" ; extension constraint
    Returns:
        parsed_constraint: dict containing the parsed constraint
    """
    parsed_constraint = {}
    variables = constraint.find("list").text.strip().split()
    parsed_constraint["variables"] = variables
    tuples = ast.literal_eval(constraint.find("conflicts").text.strip().replace(")", "),"))
    parsed_constraint["tuples"] = tuples

    return parsed_constraint


if __name__ == "__main__":
    # TODO: add tests for extension constraints
    from variables import *
    import xml.etree.ElementTree as ET
    file_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Graph-Representation\sample_problems\sample_problem_test\AircraftLanding-table-airland01_mc22.xml"
    # root = ET.parse(file_path)
    # variables = root.findall("variables")
    # array_vars = variables[0].findall("array")
    # integer_vars = variables[0].findall("var")
    # parsed_array_variables = parse_array_variables(array_vars)
    # constraints = root.findall("constraints")
    # all_different_constraints = constraints[0].findall("allDifferent")
    test_cases = ["""<allDifferent id="c1">
        <list> x1 x2 x3 x4 </list>
        <list> y1 y2 y3 y4 </list>
    </allDifferent>""",
                """<allDifferent id="c2">
        <list> v1 v2 v3 v4 </list>
        <list> w1 w2 w3 w4 </list>
        <list> z1 z2 z3 z4 </list>
        <except> (0,0,0,0) </except>
    </allDifferent>""",
     """<allDifferent>
       x1 x2 x3 x4 x5
     </allDifferent>
     """,
                """
    <allDifferent>
    <list> y[] </list>
    <except> 0 </except>
    </allDifferent>
    """,
                """<allDifferent>
    <matrix> 
        (x1,x2,x3,x4,x5)
        (y1,y2,y3,y4,y5)
        (z1,z2,z3,z4,z5)
    </matrix>
    </allDifferent>"""
    ]
    test_ad_constraints = [ET.fromstring(i) for i in test_cases]
    parse_all_different_constraints(test_ad_constraints)