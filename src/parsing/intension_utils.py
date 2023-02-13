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
    "abs",
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "sqr",
    "pow",
    "min",
    "max",
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
    "xor",
    "iff",
    "imp"
}
