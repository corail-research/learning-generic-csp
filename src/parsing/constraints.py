def parse_instance_constraints(raw_constraints):
    constraints = {}


def parse_all_different_constraints(raw_alldiff_constraints):
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


def parse_list_and_exceptions(root):
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


def parse_matrix(root):
    matrix = {}
    for matrix_element in root.findall("matrix"):
        matrix["matrix"] = [tuple(row.strip().strip("()").split(
            ",")) for row in matrix_element.text.strip().split("\n")]
    return matrix
        

def parse_base_alldiff_constraint(root):
    constraint = {}
    constraint["lists"] = root.text.strip().split()
    constraint["exceptions"] = None

    return constraint

if __name__ == "__main__":
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