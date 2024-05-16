import os
import shutil

def copy_dataset(source_folder, destination_folder, names_to_copy):
    """
    Copy folders and their contents from source_folder to destination_folder based on the presence of their names in names_to_copy.

    Args:
    - source_folder (str): Path to the source folder.
    - destination_folder (str): Path to the destination folder.
    - names_to_copy (list): List of folder names to copy.
    """

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # Iterate over folder names
    for item in names_to_copy:
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        # If the item is a file and its name is in names_to_copy, copy it to destination folder
        if os.path.isfile(str(source_path)+".pkl") and item in names_to_copy:
            shutil.copy2(str(source_path)+".pkl", destination_folder)
        # If the item is a folder and its name is in names_to_copy, copy it recursively to destination folder
        elif os.path.isdir(source_path) and item in names_to_copy:
            # Copy folder and its contents recursively
            shutil.copytree(source_path, destination_path)


# usage
source_folder = "/home/vgrwbx/workspace/OL_trajectoryprediction/data/val/raw"
destination_folder = "/home/vgrwbx/workspace/OL_trajectoryprediction/data/missingolval/raw"

ids_to_keep = set() 
# # Read IDs from file
with open('/home/vgrwbx/workspace/OL_trajectoryprediction/utils/missing_ids.csv', 'r') as file:
    next(file)  # skip the header line
    for line in file:
        parts = line.strip().split(',')
        ids_to_keep.add(parts[0])  # ID is the first element in each line

print(len(ids_to_keep))

copy_dataset(source_folder, destination_folder, ids_to_keep)

source_folder = "/home/vgrwbx/workspace/OL_trajectoryprediction/data/val/processed"
destination_folder = "/home/vgrwbx/workspace/OL_trajectoryprediction/data/missingolval/processed/"
copy_dataset(source_folder, destination_folder, ids_to_keep)
