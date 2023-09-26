import os

# Directory 1 path
directory1 = r'/scratch1/boileo/graph_coloring/data/gc_specific'

# Directory 2 path
directory2 = r'~/Graph-Representation/src/models/graph_coloring/data/gc_specific'


# Step 1: Find the maximum [number] in directory 1
max_number = 0

for filename in os.listdir(directory1):
    if filename.endswith('.graph'):
        try:
            number = int(filename[1:-6])  # Extract the [number] from the filename
            max_number = max(max_number, number)
        except ValueError:
            pass

# Step 2: Rename files in directory 2
for filename in os.listdir(directory2):
    if filename.endswith('.graph'):
        try:
            number = int(filename[1:-6])  # Extract the [number] from the filename
            new_number = max_number + number + 1
            new_filename = 'm{:07d}.graph'.format(new_number)
            os.rename(os.path.join(directory2, filename), os.path.join(directory2, new_filename))
        except ValueError:
            pass
