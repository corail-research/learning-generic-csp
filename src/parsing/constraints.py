import ast
from typing import List, Dict
import xml.etree.ElementTree as ET
from intension_utils import *

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
            constraint_type, new_constraint = parse_extension_constraint(child, instance_variables)
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
        constraint_type, new_constraint = parse_extension_group(group, instance_variables)
    elif group.find("intension") is not None:
        constraint_type, new_constraint = parse_intension_group(group)
    elif group.find("element") is not None:
        constraint_type, new_constraint = parse_element_group(group, instance_variables)
    elif group.find("allDifferent") is not None:
        constraint_type, new_constraint = parse_alldiff_group(group, instance_variables)
    elif group.find("sum") is not None:
        constraint_type, new_constraint = parse_sum_group(group, instance_variables)
    
    return constraint_type, new_constraint


def parse_extension_constraint(constraint: ET.Element, instance_variables) -> Dict:
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
    
    supports = constraint.find("supports")
    if supports is not None:
        supports = supports.text
        new_supports = parse_extension_tuple(supports, instance_variables)
        parsed_constraint["supports"] = new_supports
    
    conflicts = constraint.find("conflicts")
    if conflicts is not None:
        conflicts = conflicts.text
        new_conflicts = parse_extension_tuple(conflicts, instance_variables)
        parsed_constraint["conflicts"] = new_conflicts
    
    return "extension", parsed_constraint

def parse_extension_group(constraint: ET.Element, instance_variables) -> Dict:
    """Parse an individual extension constraint defined with supports; that is values the variables CAN take. 
    Extension constraints defined with exclusion are parsed with function parse_negative_extension_constraint
    Returns a dict like:
    {"variables": [x, y], "tuples": ((1, 2), (2, 3))}

    Args:
        constraint (ET.Element): element of "extension" ; extension constraint

    Returns:
        parsed_constraint: dict containing the parsed constraint
    """
    parsed_constraints = []
    
    # Get the support tuples
    support_tuples = []
    supports = constraint.find('extension/supports')
    if supports is not None:
        supports = supports.text
        new_supports = parse_extension_tuple(supports, instance_variables)
        support_tuples = new_supports
        
    # Get the conflict tuples
    conflict_tuples = []
    conflicts = constraint.find('extension/conflicts')
    if conflicts is not None:
        conflicts = conflicts.text
        new_conflicts = parse_extension_tuple(conflicts, instance_variables)
        conflict_tuples = new_conflicts
    
    for arg in constraint.findall('args'):
        all_arg_vars = []
        for var_string in arg.text.split():
            new_vars = parse_arg_variables(var_string, instance_variables)
            all_arg_vars.extend(new_vars)
        parsed_constraints.append({"variables": all_arg_vars, "supports": support_tuples, "conflicts": conflict_tuples})
    
    return "extension", parsed_constraints

def parse_extension_tuple(raw_tuple: str, instance_variables) -> List:
    """Parse a support or conflict tuple"""
    if "(" in raw_tuple:
        tuples = ast.literal_eval(raw_tuple.strip().replace("*", "None").replace(")", "),"))
    else:
        tuples = [[int(i)] for i in parse_variable_domain(raw_tuple)]
    
    return tuples

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

    # file_path = r"C:/Users/leobo/Desktop/École/Poly/SeaPearl/instancesXCSP22/MiniCOP/LowAutocorrelation-015_c18.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCSP\Rlfap-ext-scen-11-f12_c18.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCSP\NumberPartitioning-290_mc22.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCOP\ClockTriplet-20-35_c22.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\SeaPearl\instancesXCSP22\MiniCSP\Ortholatin-20_mc22.xml..xml"
    # file_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCOP23\TSPTW-n020w040-1_mini_c23.xml"
    file_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\XCSP23_V2\MiniCSP23\CoveringArray-3-05-2-10_mini_c23.xml"
    root = ET.parse(file_path)
    variables = root.findall("variables")
    instance_variables = parse_all_variables(variables)
    constraints = root.findall("constraints")[0]
    constraints = parse_constraint_section(instance_variables, constraints)
    a=1
