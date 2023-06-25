import os
import re

unwanted_extensions = [
    ".DS_Store",
    ".pyc",
    ".pth",
    ".gz",
]  # Add any other unwanted file extensions


def test_unwanted_files():
    root_dir = ".."  # Replace with the root directory of your repository
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(filenames)
        print(dirpath)
        if "tests" in re.split(r"[\\/]", dirpath):
            print(dirpath)
            continue  # Skip the current directory
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1]
            assert (
                file_extension not in unwanted_extensions
            ), f"Unwanted file found: {filename}"
