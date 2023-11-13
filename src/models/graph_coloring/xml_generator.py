import os

def parse_graph_coloring_instance(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    dimension_line = next(line for line in lines if line.startswith('DIMENSION'))
    dimension = int(dimension_line.split(':')[1].strip())

    edges = []
    for line in lines[lines.index('EDGE_DATA_SECTION\n')+1:]:
        if line.strip() == '-1':
            break
        edges.append(tuple(map(int, line.split())))

    diff_edge_index = next(i for i, line in enumerate(lines) if line.startswith('DIFF_EDGE'))
    diff_edge = tuple(map(int, lines[diff_edge_index + 1].split()))

    chrom_number_index = next(i for i, line in enumerate(lines) if line.startswith('CHROM_NUMBER'))
    chrom_number = int(lines[chrom_number_index + 1].strip())
    return edges, diff_edge, chrom_number, dimension

def save_graph_coloring_instance_to_file(filepath, graph, chrom_number, dimension, instance_number, variant):
    filename = f"data{instance_number}_{variant}.xml"
    filepath = os.path.join(filepath, filename)
    with open(filepath, 'w') as file:
        file.write('<instance format="XCSP3" type="COP">\n')
        file.write('  <variables>\n')
        for node in range(dimension):
            file.write(f'    <var id="x{node}"> 1..{chrom_number} </var>\n')
        file.write('  </variables>\n')
        file.write('  <constraints>\n')
        for edge in graph:
            if edge:
                file.write(f'    <intension> ne(x{edge[0]},x{edge[1]}) </intension>\n')
        file.write('  </constraints>\n')
        file.write('</instance>\n')

def build_graph_coloring_instances(directory, save_path='graph_coloring_instances'):
    # Ensure there's a directory to save the instances
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Loop over all files in the directory
    for instance_number, filename in enumerate(os.listdir(directory)):
        filepath = os.path.join(directory, filename)

        # Parse the instance and create the XCSP3 instances
        edges, diff_edge, chrom_number, dimension = parse_graph_coloring_instance(filepath)
        save_graph_coloring_instance_to_file(save_path, edges, chrom_number, dimension, instance_number, 0)
        save_graph_coloring_instance_to_file(save_path, edges + [diff_edge], chrom_number, dimension, instance_number, 1)

if __name__ == "__main__":
    directory = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\graph_coloring\data\raw"
    build_graph_coloring_instances(directory)

