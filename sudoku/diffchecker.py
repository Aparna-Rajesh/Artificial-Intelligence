import difflib
from distutils.log import error

with open('output.txt') as file_1:
    file_1_text = file_1.readlines()

with open('sudokus_finish.txt') as file_2:
    file_2_text = file_2.readlines()

# Find and print the diff:

error_found = False
for line in difflib.unified_diff(
        file_1_text, file_2_text, fromfile='output.txt',
        tofile='sudokus_finish.txt', lineterm=''):
    if line != "":
        error_found = True
    print(line)

if not error_found:
    print("No discrepancies found")
