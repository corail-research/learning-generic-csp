#!/bin/bash

# List of dictionaries, each containing the arguments for a particular training configuration
configs=(
    # Configuration 1
    '{"lr": 0.001, "batch_size": 32, "num_layers": 4, "num_heads": 8, "num_epochs": 100, "hidden_size": 512, "train_bounds": [0, 1000], "valid_bounds": [1000, 1100], "dropout": 0.1}',
    # Configuration 2
    '{"lr": 0.0001, "batch_size": 64, "num_layers": 6, "num_heads": 12, "num_epochs": 50, "hidden_size": 1024, "train_bounds": [0, 2000], "valid_bounds": [2000, 2100], "dropout": 0.2}'
)

# Path to the data directory
path="./data"

# Device to use for training
device="cuda:0"

# Parse command-line arguments (if any)
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -path)
    path="$2"
    shift
    shift
    ;;
    -device)
    device="$2"
    shift
    shift
    ;;
    *)
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

# Loop over the list of training configurations and train the model for each one
for config in "${configs[@]}"
do
    # Parse the JSON string into a dictionary and update the default values
    declare -A args=$(echo "$config" | jq -r 'to_entries | map("\(.key)=\(.value|tostring)") | join(" ")')
    lr=${args[lr]:-0.001}
    batch_size=${args[batch_size]:-32}
    num_layers=${args[num_layers]:-4}
    num_heads=${args[num_heads]:-8}
    num_epochs=${args[num_epochs]:-100}
    hidden_size=${args[hidden_size]:-512}
    train_start=${args[train_bounds]:-0}
    train_end=${args[train_bounds]:-1000}
    valid_start=${args[valid_bounds]:-1000}
    valid_end=${args[valid_bounds]:-1100}
    dropout=${args[dropout]:-0.1}

    # Train the model using the provided arguments
    python train.py \
        -path "$path" \
        -lr "$lr" \
        -batch_size "$batch_size" \
        -num_layers "$num_layers" \
        -num_heads "$num_heads" \
        -num_epochs "$num_epochs" \
        -hidden_size "$hidden_size" \
        -device "$device" \
        -train_bounds "$train_start" "$train_end" \
        -valid_bounds "$valid_start" "$valid_end" \
        -dropout "$dropout"
done
