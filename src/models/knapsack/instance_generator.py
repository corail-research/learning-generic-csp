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

def generate_knapsack_instance(n_items, max_weight, max_value):
    weights = [random.randint(1, max_weight) for _ in range(n_items)]
    values = [random.randint(1, max_value) for _ in range(n_items)]
    W = random.randint(n_items, n_items * max_weight)
    return weights, values, W

def save_instance_to_file(filepath, instance, index):
    filename = f"instance_{index}.json"
    filepath = os.path.join(filepath, filename)
    with open(filename, 'w') as file:
        json.dump(instance, file, indent=4)

def build_knapsack_dataset(n_instances, n_items, max_weight, max_value, save_path='knapsack_instances'):
    # Ensure there's a directory called 'knapsack_instances' to save the instances
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(n_instances):
        weights, values, W = generate_knapsack_instance(n_items, max_weight, max_value)
        optimal_value = knapsack(weights, values, W, n_items)
        instance = {
            "weights": weights,
            "values": values,
            "W": W,
            "optimal_value": optimal_value
        }
        save_instance_to_file(save_path, instance, i+1)

# Parameters
N_INSTANCES = 100
N_ITEMS = 10
MAX_WEIGHT = 50
MAX_VALUE = 100

build_knapsack_dataset(N_INSTANCES, N_ITEMS, MAX_WEIGHT, MAX_VALUE)
