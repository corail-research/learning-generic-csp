import random
import json
import os

def knapsack(weights, values, W, n):
    dp = [[0 for w in range(W+1)] for i in range(n+1)]

    for i in range(n+1):
        for w in range(W+1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][W]

def generate_knapsack_instance(n_items_range, max_weight, max_value):
    n_items = random.randint(n_items_range[0], n_items_range[1])
    weights = [random.randint(1, max_weight) for _ in range(n_items)]
    values = [random.randint(1, max_value) for _ in range(n_items)]
    W = random.randint(n_items, n_items * max_weight)
    return weights, values, W, n_items

def build_knapsack_dataset(n_instances, n_items_range, max_weight, max_value, save_path='knapsack_instances'):
    # Ensure there's a directory called 'knapsack_instances' to save the instances
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    while i < n_instances:
        weights, values, W, n_items = generate_knapsack_instance(n_items_range, max_weight, max_value)
        optimal_value = knapsack(weights, values, W, n_items)
        if optimal_value > 0:
            instance = {
                "weights": weights,
                "values": values,
                "W": W,
                "optimal_value": optimal_value
            }
            save_instance_to_file(save_path, instance, i + 1)
            i += 1

def save_instance_to_file(filepath, instance, index):
    filename = f"instance_{index}.xml"
    filepath = os.path.join(filepath, filename)
    with open(filepath, 'w') as file:
        file.write('<instance format="XCSP3" type="COP">\n')
        file.write('  <variables>\n')
        file.write(f'    <array id="x" size="[{len(instance["weights"])}]"> 0..1 </array>\n')
        file.write('  </variables>\n')
        file.write('  <constraints>\n')
        file.write('    <sum id="c1">\n')
        file.write(f'      <list> {" ".join([f"x[{i}]" for i in range(len(instance["weights"]))])} </list>\n')
        file.write(f'      <coeffs> {" ".join(map(str, instance["weights"]))} </coeffs>\n')
        file.write(f'      <condition> (le,{instance["W"]}) </condition>\n')
        file.write('    </sum>\n')
        file.write('  </constraints>\n')
        file.write('  <objectives>\n')
        file.write('    <minimize type="sum">\n')
        file.write(f'      <list> {" ".join([f"x[{i}]" for i in range(len(instance["values"]))])} </list>\n')
        file.write(f'      <coeffs> {" ".join(map(str, instance["values"]))} </coeffs>\n')
        file.write('    </minimize>\n')
        file.write(f'    <optimal>{instance["optimal_value"]}</optimal>\n')
        file.write('  </objectives>\n')
        file.write('</instance>\n')


# Parameters
N_INSTANCES = 10000
N_ITEMS = (20, 40)
MAX_WEIGHT = 50
MAX_VALUE = 100
import time
start = time.time()
build_knapsack_dataset(N_INSTANCES, N_ITEMS, MAX_WEIGHT, MAX_VALUE)
end = time.time()
print(f"Time elapsed: {end-start} seconds")
#time per instance
print(f"Time per instance: {(end-start)/N_INSTANCES} seconds")