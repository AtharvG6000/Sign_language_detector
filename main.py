import os
import random

# Define the path to the root directory containing the 23 subdirectories
root_dir = "./data"

# Desired number of files to retain in each directory
files_to_retain = 500

# Iterate through each subdirectory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        # Get a list of all .jpg files in the subdirectory
        jpg_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')]
        
        # Check if there are more files than the desired number to retain
        if len(jpg_files) > files_to_retain:
            # Calculate the number of files to delete
            files_to_delete = len(jpg_files) - files_to_retain
            
            # Randomly select files to delete
            files_to_remove = random.sample(jpg_files, files_to_delete)
            for file_name in files_to_remove:
                file_path = os.path.join(subdir_path, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            print(f"{subdir_path} already has {len(jpg_files)} files or fewer.")
