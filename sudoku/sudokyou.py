#!/usr/bin/env python
# coding:utf-8
"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""
import sys
import copy
from time import time
import numpy as np

ROW = "ABCDEFGHI"
COL = "123456789"

rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9']


class csp:
    def __init__(self, config):  # config is just a string of all the values that the user inputted
        self.arr, self.dict, self.set = [], dict(), set()
        print(config)
        self.init(config)

    def init(self, config):
        for i in range(9):
            for j in range(0, 9):  # same exact thing as range(9)
                index = (i*9)+j
                self.arr.append(rows[i]+cols[j])
                if config[index] == '0':
                    domain = COL
                else:
                    domain = config[index]
                self.dict[rows[i]+cols[j]] = domain


def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)


def backtracking(config):
    """Takes a board and returns solved board."""
    domains = dict()
    return solve(config, domains)


def solution(config, domains):
    for item in config.arr:
        if item not in domains:
            return False
    return True


def neighbors(config, key):
    index = config.arr.index(key)
    row, col = int(index/9), index % 9
    neighbor = set()
    for i in range(1, 10):
        newRow, newCol = rows[row] + str(i), rows[i-1] + str(col+1)
        if newRow != key:
            neighbor.add(newRow)
        if newCol != key:
            neighbor.add(newCol)
    currentRow, currentCol = int(row/3)*3, int(col/3)*3
    for i in range(currentRow, currentRow+3):
        for j in range(currentCol, currentCol+3):
            newBox = rows[i] + str(j+1)
            if newBox != key:
                neighbor.add(newBox)
    return neighbor


def solve(config, domains):  # config is the cst object, and domains is all of the assigned values

    if solution(config, domains):
        return domains
    ans = None
    key = min([domain for domain in config.dict if domain not in domains],
              key=lambda Key: len(config.dict[Key]))  # MRV Heuristic
    for value in config.dict[key]:
        consistent = True
        for neighbor in neighbors(config, key):
            if neighbor in domains and domains[neighbor] == value:
                consistent = False
                break
        if consistent == True:
            newconfig = copy.deepcopy(config)
            newconfig.dict[key], domains[key] = value, value
            res = True
            # Forward Checking:
            for neighbor in neighbors(newconfig, key):
                if neighbor not in domains:
                    newconfig.dict[neighbor] = newconfig.dict[neighbor].replace(
                        value, '')
                    if len(newconfig.dict[neighbor]) > 0:
                        continue
                    else:
                        res = False
                        break
            if res == True:
                ans = solve(newconfig, domains)
                if ans is not None:
                    return ans
        del domains[key]  # Backtracking step
    return ans


if __name__ == '__main__':
    if len(sys.argv) > 1:

        # Running sudoku solver with one board $python3 sudoku.py <input_string>.
        print(sys.argv[1])
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = {ROW[r] + COL[c]: int(sys.argv[1][9*r+c])
                 for r in range(9) for c in range(9)}

        solved_board = backtracking(csp(sys.argv[1]))
        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(board_to_string(solved_board))
        outfile.write('\n')
    else:
        # Running sudoku solver for boards in sudokus_start.txt $python3 sudoku.py
        #  Read boards from source.
        src_filename = 'sudokus_start.txt'
        try:
            srcfile = open(src_filename, "r")
            sudoku_list = srcfile.read()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()
        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        timings = []
        # Solve each board using backtracking
        for line in sudoku_list.split("\n"):
            start = time()
            if len(line) < 9:
                continue
            #print("line: " + line)
            # Parse boards to dict representation, scanning board L to R, Up to Down
            # board = { ROW[r] + COL[c]: int(line[9*r+c])
            #           for r in range(9) for c in range(9)}
            # # Print starting board. TODO: Comment this out when timing runs.
            # print_board(board)
            # Solve with backtracking
            # print("THE CSP")
            # print(csp(line).dict)
            solved_board = backtracking(csp(line))
            end = time()
            timings.append(end - start)
            # Print solved board. TODO: Comment this out when timing runs.
            print_board(solved_board)
            # Write board to file
            outfile.write(board_to_string(solved_board))
            outfile.write('\n')

        timings = np.array(timings)
        minTime = np.min(timings)
        maxTime = np.max(timings)
        avgTime = np.mean(timings)
        stDev = np.std(timings)
        board_solved = len(timings)
        with open("README.txt", "w") as fp:
            fp.write('Boards Solved: {}\n'.format(board_solved))
            fp.write('Minimum Runtime: {}\n'.format(minTime))
            fp.write('Maximum Runtime: {}\n'.format(maxTime))
            fp.write('Mean Runtime: {}\n'.format(avgTime))
            fp.write('Standard Deviation: {}\n'.format(stDev))
        print("Finishing all boards in file.")
