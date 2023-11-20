import re

class IntensionNode:
    def __init__(self, operator, children):
        self.operator = operator
        self.children = children


def replace_placeholders(string, args):
    for i, arg in enumerate(args):
        placeholder = f"%{i}"
        string = string.replace(placeholder, str(arg))
    return string

intension_operators = {
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "dist",
    "lt",
    "le",
    "ge",
    "gt",
    "ne",
    "eq",
    "not",
    "and",
    "or",
}

def parse_intension_group(group):
    """Parses a group of intension constraints in a given problem

    Args:
        group (xml.etree.ElementTree.Element): An XML element representing the group of intension constraints
        base_intension_expression (str): The base intension expression to use for parsing the constraints

    Returns:
        List[IntensionNode]: A list of IntensionNode objects representing the parsed intension constraints
    """
    base_intension_expression = group.find("intension").text.strip()
    parsed_intension_constraints = []
    args = group.findall("args")
    for raw_arg in args:
        arg_format = raw_arg.text.strip().split()
        formatted = replace_placeholders(base_intension_expression, arg_format)
        parsed = parse_intension_constraint(formatted)
        parsed_intension_constraints.append(parsed)
    
    return "intension", parsed_intension_constraints

def parse_intension_constraint(expression):
    """Parses an intension expression in a given problem

    Args:
        expression (str): The intension expression to parse

    Returns:
        Union[str, IntensionNode]: Either a string representing the parsed expression, or an IntensionNode object representing the parsed expression
    """
    # Split the expression into operator and operands
    if type(expression) is not str:
        expression = expression.text.strip()
    split = re.split(r"\((.*)\)", expression)
    operator = split[0]
    if operator not in intension_operators:
        # If the expression does not have any further parentheses, return it as a string
        return expression
    operands_str = split[1]
    # Recursively parse the operands
    operands = []
    start = 0
    balance = 0
    for i, char in enumerate(operands_str):
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        elif char == "," and balance == 0:
            operands.append(parse_intension_constraint(operands_str[start:i]))
            start = i + 1
    operands.append(parse_intension_constraint(operands_str[start:]))

    # Return the IntensionNode for the expression
    return IntensionNode(operator, operands)