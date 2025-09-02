"""
File System.
"""
import os

def new_dir(folder_path):
    """
    Construct a new folder.
    """
    if os.path.exists(folder_path):
        print(f"{folder_path} has already existed!")
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")

