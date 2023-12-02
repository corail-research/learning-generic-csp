try:
    from generic_xcsp.variable_parsing import *
except:
    from variable_parsing import *
from typing import List, Dict
import xml.etree.ElementTree as ET


def parse_element_constraint(constraint: ET.Element, instance_variables: Dict) -> Dict:
    """Parses an element constraint. Returns a dict with the parsed constraint"""
    elm_list_text = constraint.find("list").text
    elm_list = elm_list_text.strip().split(" ")
    elm_list_contains_variables = False
    if instance_variables.contains_variable(elm_list[0]):
        elm_list = parse_arg_variables(elm_list_text, instance_variables)
        elm_list_contains_variables = True
    else:
        elm_list = [float(elm) for elm in elm_list]
    # value_position = int(constraint.find("value").text.strip().replace("%", ""))
    index_position = None
    index_element = constraint.find("index")
    if index_element is not None:
        index_text = index_element.text.strip()
        if instance_variables.contains_variable(index_text):
            index_position = index_text
        else:
            index_position = int(index_text)
    value_position = None
    value_element = constraint.find("value")
    if value_element is not None:
        value_text = value_element.text.strip()
        if instance_variables.contains_variable(value_text):
            value_position = value_text
        else:
            value_position = int(value_text)
            
    return "element", {"list": elm_list, "index": index_position, "value": value_position, "list_contains_variables": elm_list_contains_variables}
    

def parse_element_group(group: ET.Element, instance_variables: Dict) -> Dict:
    """Parses an element group. Returns a dict with the parsed constraint"""
    parsed_group = []
    elm_list_contains_variables = False
    elm_element = group.find("element")
    elm_list_text = elm_element.find("list").text
    elm_list = elm_list_text.strip().split(" ")
    if instance_variables.contains_variable(elm_list[0]):
        elm_list = parse_arg_variables(elm_list_text, instance_variables)
        elm_list_contains_variables = True
    else:
        elm_list = [float(elm) for elm in elm_list]
    
    value_position = int(elm_element.find("value").text.strip().replace("%", ""))
    index_position = None
    index_element = elm_element.find("index")

    if index_element is not None:
        index_text = index_element.text.strip().replace("%", "")
        if instance_variables.contains_variable(index_text):
            index_position = parse_arg_variables(index_text, instance_variables)
        else:
            index_position = int(index_text)    
    
    args_elements = group.findall("args")
    for args_element in args_elements:
        args_text = args_element.text.strip().split(" ")
        value_element = args_text[value_position]
        if index_position is not None:
            index_element = args_text[index_position]
        else:
            index_element = None
        parsed_sum = {"list": elm_list, "index": index_element, "value": value_element, "list_contains_variables": elm_list_contains_variables}
        parsed_group.append(parsed_sum)

    return "element", parsed_group

