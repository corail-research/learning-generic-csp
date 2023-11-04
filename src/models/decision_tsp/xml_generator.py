import re
import xml.etree.ElementTree as ET

def parse_tsp_data(input_file):
    """
    Parses the TSP data file and returns a matrix of distances between cities.
    """
    with open(input_file, 'r') as file:
        content = file.read()

    # Extract the EDGE_WEIGHT_SECTION part
    weights_section = re.search(r"EDGE_WEIGHT_SECTION:([\s\S]*?)\nTOUR_SECTION:", content)
    if not weights_section:
        raise ValueError("EDGE_WEIGHT_SECTION not found in the file.")

    weights_text = weights_section.group(1).strip()
    weights_matrix = [list(map(float, line.split())) for line in weights_text.split('\n')]
    
    optimal_value_section = re.search(r"OPTIMAL_VALUE:\n([\d\.]+)", content)
    if not optimal_value_section:
        raise ValueError("OPTIMAL_VALUE not found in the file.")

    optimal_value = float(optimal_value_section.group(1))

    return weights_matrix, optimal_value


def add_tail(element):
    """Adds a newline as a tail to each XML element for better formatting."""
    element.tail = '\n'
    for child in element:
        add_tail(child)

def create_xml_instance(weights_matrix, optimal_value, output_file):
    """
    Creates an XML instance for the TSP problem based on the weights matrix.
    """
    num_cities = len(weights_matrix)
    # XML Root
    root = ET.Element("instance", format="XCSP3", type="COP")
    
    # Variables
    variables = ET.SubElement(root, "variables")
    cities_array = ET.SubElement(variables, "array", id="x", size=str(num_cities))
    # Define domains for x
    for i in range(num_cities):
        domain = ET.SubElement(cities_array, "domain", {"for": f"x[{i}]"})
        domain.text = " " + " ".join(str(j) for j in range(num_cities) if j != i) + " "

    distances_array = ET.SubElement(variables, "array", id="d", type="real", size=str(num_cities))
    for i in range(num_cities):
        domain = ET.SubElement(distances_array, "domain", {"for": f"d[{i}]"})
        domain.text = " " + " ".join(str(max(weights_matrix[i][j], weights_matrix[j][i])) for j in range(num_cities) if i != j)

    # Constraints
    constraints = ET.SubElement(root, "constraints")
    for i in range(num_cities):
        extension = ET.SubElement(constraints, "extension")
        supports_list = ET.SubElement(extension, "list")
        supports_list.text = f" x[{i}] d[{i}] "
        supports = ET.SubElement(extension, "supports")
        supports_text = " ".join(f"({j},{max(weights_matrix[i][j], weights_matrix[j][i])})" for j in range(num_cities) if i != j)
        supports.text = supports_text

    # AllDifferent Constraint
    alldiff = ET.SubElement(constraints, "allDifferent")
    alldiff.text = " x[] "

    # Objective
    objectives = ET.SubElement(root, "objectives")
    minimize = ET.SubElement(objectives, "minimize")
    minimize.text = " sum(d[]) "
    optimal = ET.SubElement(objectives, "optimal")
    optimal.text = str(optimal_value)

    # Formatting
    add_tail(root)

    # Writing to the file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

def main(input_file, output_file):
    weights_matrix, optimal_value = parse_tsp_data(input_file)
    create_xml_instance(weights_matrix, optimal_value, output_file)
    print(f"XML instance created: {output_file}")


if __name__ == "__main__":
    input_file = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\decision_tsp\data\raw\0.graph"
    output_file = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\decision_tsp\text2.xml"
    main(input_file, output_file)