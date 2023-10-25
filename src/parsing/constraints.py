import ast
from typing import List, Dict
import xml.etree.ElementTree as ET
from intension_utils import *
import extension
from variables import *
from alldifferent import *
from element import *
from sum_constraint import *


def parse_constraint_section(instance_variables:Dict, block: ET.Element, constraints={}) -> List[Dict]:
    """Parses the constraint section. Works only for extension constraints. Returns a list of dicts from the parse_extension_constraint
    function

    Args:
        block: element of name "block". Contains constraints grouped together

    Returns:
        constraints (List[Dict]): parsed extension constraints
    """
    block = list(block)
    for child in block:
        name = child.tag
        if name == "block":
            constraints.update(parse_constraint_section(instance_variables, child, constraints))
        
        elif name == "group":
            constraint_type, new_constraint = parse_group(child, instance_variables)
            if constraints.get(constraint_type) is None:
                constraints[constraint_type] = []
            constraints[constraint_type].extend(new_constraint)
        
        elif name == "extension":
            constraint_type, new_constraint = parse_extension_constraint(child)
            if constraints.get(constraint_type) is None:
                constraints[constraint_type] = [new_constraint]
            else:
                constraints[constraint_type].append(new_constraint)

        elif name == "intension":
            constraint_type = "intension"
            new_constraint = parse_intension_constraint(child)
            if constraints.get(constraint_type) is None:
                constraints[constraint_type] = [new_constraint]
            else:
                constraints[constraint_type].append(new_constraint)
        
        elif name == "allDifferent":
            constraint_type, new_constraint = parse_alldiff_constraint(child, instance_variables)
            contains_multiple_constraints = type(new_constraint[0]) == list 
            if constraints.get(constraint_type) is None:
                if contains_multiple_constraints:
                    constraints[constraint_type] = new_constraint
                else:
                    constraints[constraint_type] = [new_constraint]
            else:
                if contains_multiple_constraints:
                    constraints[constraint_type].extend(new_constraint)
                else:
                    constraints[constraint_type].append(new_constraint)
        
        elif name == "sum":
            constraint_type, new_constraint = parse_sum(child, instance_variables)
            if constraints.get(constraint_type) is None:
                constraints[constraint_type] = [new_constraint]
            else:
                constraints[constraint_type].append(new_constraint)

        elif name == "element":
            constraint_type, new_constraint = parse_element(child, instance_variables)
            if constraints.get(constraint_type) is None:
                constraints[constraint_type] = [new_constraint]
            else:
                constraints[constraint_type].append(new_constraint)

    return constraints

def parse_group(group: ET.Element, instance_variables:Dict) -> List[Dict]:
    """Parse a group block: Identifies the type of constraint and parses it accordingly
    """
    constraint_type, new_constraint = None, None
    if group.find("extension") is not None:
        constraint_type, new_constraint = parse_extension_group(group)
    elif group.find("intension") is not None:
        constraint_type, new_constraint = parse_intension_group(group)
    elif group.find("element") is not None:
        pass
    elif group.find("allDifferent") is not None:
        constraint_type, new_constraint = parse_alldiff_group(group, instance_variables)
    elif group.find("sum") is not None:
        constraint_type, new_constraint = parse_sum_group(group, instance_variables)
    
    return constraint_type, new_constraint


def parse_extension_constraint(constraint: ET.Element) -> Dict:
    """Parse basic extension constraint and return it as a dict
    ex:
    <extension id="c1">
        <list> x1 x2 x3 </list> 
        <supports> (0,1,0) (1,0,0) (1,1,0) (1,1,1) </supports>
    </extension>
    will return:
    {
        "variables": [x1, x2, x3],
        "supports": [(0,1,0), (1,0,0), (1,1,0), (1,1,1)]
    }

    Args:
        constraint (ET.Element): _description_

    Returns:
        Dict: _description_
    """
    parsed_constraint = {}
    variables = constraint.find("list").text.split()
    parsed_constraint["variables"] = variables
    
    support_tuples = []
    supports = constraint.find("supports")
    if supports is not None:
        supports = supports.text
        parsed_constraint["supports"] = list(ast.literal_eval(supports.replace(")", "),").strip()))
    
    
    conflicts = constraint.find("conflicts")
    if conflicts is not None:
        conflicts = conflicts.text
        parsed_constraint["conflicts"] = list(ast.literal_eval(conflicts.replace(")", "),")))
    
    return "extension", parsed_constraint

def parse_extension_group(constraint: ET.Element) -> Dict:
    """Parse an individual extension constraint defined with supports; that is values the variables CAN take. 
    Extension constraints defined with exclusion are parsed with function parse_negative_extension_constraint
    Returns a dict like:
    {"variables": [x, y], "tuples": ((1, 2), (2, 3))}

    Args:
        constraint (ET.Element): element of "extension" ; extension constraint

    Returns:
        parsed_constraint: dict containing the parsed constraint
    """
    parsed_constraint = {}

    variables = []
    for arg in constraint.findall('args'):
        variables.append((arg.text.split()[0], arg.text.split()[1]))
    parsed_constraint["variables"] = variables
    
    # Get the support tuples
    support_tuples = []
    supports = constraint.find('extension/supports')
    if supports is not None:
        supports = supports.text.split()
        for support in supports:
            support_tuples.append(ast.literal_eval(support.replace(")", "),")))
        parsed_constraint["supports"] = support_tuples
        
    # Get the conflict tuples
    conflict_tuples = []
    conflicts = constraint.find('extension/conflicts')
    if conflicts is not None:
        conflicts = conflicts.text.split()
        for conflict in conflicts:
            conflict_tuples.extend(ast.literal_eval(conflict.replace(")", "),")))
        parsed_constraint["conflicts"] = conflict_tuples

    return "extension", parsed_constraint

if __name__ == "__main__":
    # TODO: add tests for extension constraints
    from variables import *
    import xml.etree.ElementTree as ET
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCOP\AircraftLanding-table-airland02_mc22.xml..xml"
    file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCOP\ClockTriplet-20-35_c22.xml..xml"
    root = ET.parse(file_path)
    variables = root.findall("variables")
    array_vars = variables[0].findall("array")
    integer_vars = variables[0].findall("var")
    parsed_array_variables = parse_array_variables(array_vars)
    constraints = root.findall("constraints")
    all_different_constraints = constraints[0].findall("allDifferent")
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
    # parse_alldiff_constraint(test_ad_constraints)

    extension_constraint = """<extension id="c1">
    <list> x1 x2 x3 </list> 
    <supports> (0,1,0) (1,0,0) (1,1,0) (1,1,1)
    </supports>
    </extension>
    """
    test_extension_constraint = ET.fromstring(extension_constraint)
    parsed_extension_constraint = parse_extension_constraint(test_extension_constraint)

    # file_path = r"C:/Users/leobo/Desktop/École/Poly/SeaPearl/instancesXCSP22/MiniCOP/LowAutocorrelation-015_c18.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCSP\Rlfap-ext-scen-11-f12_c18.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCSP\NumberPartitioning-290_mc22.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCOP\ClockTriplet-20-35_c22.xml..xml"
    file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCSP\Ortholatin-20_mc22.xml..xml"
    root = ET.parse(file_path)
    variables = root.findall("variables")
    instance_variables = parse_all_variables(variables)
    constraints = root.findall("constraints")[0]
    # groups = constraints[0].findall("group")
    # base_intension_expression = groups[0].find("intension").text.strip()
    # intension_groups = parse_intension_group(groups[0], base_intension_expression)
    groups = constraints
    # for group in groups:
    constraints = parse_constraint_section(instance_variables, constraints)
    a=1
