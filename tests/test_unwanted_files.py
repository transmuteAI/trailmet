# MIT License
#
# Copyright (c) 2023 Transmute AI Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import re

unwanted_extensions = [
    '.DS_Store',
    '.pyc',
    '.pth',
    '.gz',
]  # Add any other unwanted file extensions


def test_unwanted_files():
    root_dir = '..'  # Replace with the root directory of your repository
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(filenames)
        print(dirpath)
        if 'tests' in re.split(r'[\\/]', dirpath):
            print(dirpath)
            continue  # Skip the current directory
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1]
            assert (
                file_extension
                not in unwanted_extensions), f'Unwanted file found: {filename}'
