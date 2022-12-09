#!/usr/bin/env python
# coding:utf-8

"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8

^ I've changed this so that it is repsented as string : string
"""
import copy
import re
import sys
from time import time
import numpy as np

ROW = "ABCDEFGHI"
COL = "123456789"


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


class Board:
    def __init__(self, givendict: dict):
        # the values of this dict should hold all the domain vals.
        self.domains_dict = dict()

        for x in givendict.keys():
            if givendict.get(x) == 0:
                # if they give us a blank tile, then we should set the variable of this tile to all possible values
                self.domains_dict[x] = str(COL)  # copies
            else:
                self.domains_dict[x] = str(givendict.get(x))

        self.variables = list(givendict.keys())  # copies and becomes a list


def minimum_remaining_values(board: Board, assigned: dict) -> str:
    '''
    heuristic to grab the most constrained variable first
    '''

    # getting an array of all of the unassigned variables in the board
    unassigned_vars = []
    for x in board.domains_dict:
        if x not in assigned:
            unassigned_vars.append(x)

    low = sys.maxsize
    low_var = None
    for value in unassigned_vars:
        if len(board.domains_dict[value]) < low:
            low = len(board.domains_dict[value])
            low_var = value

    return low_var


def calculate_children(board: Board, given_var: str, row: int, col: int) -> list:
    neighbors = set()

    for i in range(9):
        new_row = ROW[row] + str(i + 1)
        new_column = ROW[i] + str(col+1)
        if new_row != given_var:
            neighbors.add(new_row)
        if new_column != given_var:
            neighbors.add(new_column)

    curr_row = (row//3)
    curr_row = curr_row * 3

    curr_column = (col//3)
    curr_column = curr_column * 3

    for i in range(curr_row, curr_row+3):
        for j in range(curr_column, curr_column+3):
            nw_square = ROW[i] + str(j+1)
            if nw_square != given_var:
                neighbors.add(nw_square)
    return neighbors


def get_children(board: Board, given_var: str) -> list:
    a = 0
    index = -1  # index of the variable we are looking for
    for x in board.variables:
        if x == given_var:
            index = a
        a = a + 1

    row = index//9
    col = index % 9

    return calculate_children(board, given_var, row, col)


def consistent(board: Board, assigned: dict, min_val: str, values: str) -> bool:
    # value is a possible value for the selected variable
    for bros in get_children(board, min_val):
        if bros in assigned and assigned[bros] == values:
            return False
    return True


def forward_checking(child_board: Board, assigned: dict, var: str, value: str) -> tuple:
    can_continue = True
    for child in get_children(child_board, var):
        if child not in assigned:
            child_board.domains_dict[child] = re.sub(
                value, '', child_board.domains_dict[child])
            if len(child_board.domains_dict[child]) > 0:
                continue
            else:
                can_continue = False
                break
    if can_continue:
        solved = solving_board(child_board, assigned)
        if solved is not None:
            return (can_continue, solved)
    return None


'''
Following prof ansaf's slides as closely as I can:

function BACKTRACK(csp, assignment) returns a solution or failure
    if assignment is complete then return assignment
    var ← SELECT-UNASSIGNED-VARIABLE(csp, assignment)
    for each value in ORDER-DOMAIN-VALUES(csp, var, assignment) do
        if value is consistent with assignment then
            add {var = value} to assignment 
            inferences←INFERENCE(csp,var,assignment) 
            if inferences ̸= failure then
                add inferences to csp
                result ← BACKTRACK(csp, assignment) 
                if result ̸= failure then return result 
                remove inferences from csp
            remove {var = value} from assignment 
    return failure

'''


def solving_board(board: Board, assigned: dict) -> dict:
    # this is the function that will actuallu solve the board
    '''
    assigned will have all of the domains that are actually solved
    '''
    # if assignment is complete then return assignment
    solved_bool = True
    for x in board.variables:
        if x not in assigned:
            solved_bool = False
            break
    if solved_bool:
        return assigned

    # var ← SELECT-UNASSIGNED-VARIABLE(csp, assignment)
    var = minimum_remaining_values(board, assigned)
    solved = None

    # for each value in ORDER-DOMAIN-VALUES(csp, var, assignment)
    for value in board.domains_dict[var]:
        is_consistent = True

        for neighbor in get_children(board, var):
            if neighbor in assigned and assigned[neighbor] == value:
                is_consistent = False
                break

        if is_consistent:  # if we found something that works!
            child_board = Board(board.domains_dict)
            child_board.domains_dict[var] = value
            assigned[var] = value

            solved = forward_checking(child_board, assigned, var, value)

            if solved is not None:
                if solved[0] and solved[1] is not None:
                    return solved[1]  # if result ̸= failure then return result
        del assigned[var]  # remove {var = value} from assignment
    return solved


'''
from the textbook/slides: 

function BACKTRACKING-SEARCH(csp) 
    returns a solution or failure return BACKTRACK(csp,{})
'''


def backtracking(board):  # this function was given. do not modify
    """Takes a board and returns solved board."""
    # TODO: implement this
    board_obj = Board(board)

    solved = solving_board(board_obj, dict())
    return solved


def write_to_file(times):
    timings = np.array(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_time = np.mean(times)
    std_deviation = np.std(times)
    num_solved = len(times)
    with open("README.txt", "w") as w:
        w.write('Boards Solved: {}\n'.format(num_solved))
        w.write('Minimum Runtime: {}\n'.format(min_time))
        w.write('Maximum Runtime: {}\n'.format(max_time))
        w.write('Mean Runtime: {}\n'.format(avg_time))
        w.write('Standard Deviation: {}\n'.format(std_deviation))


if __name__ == '__main__':
    times = []

    if len(sys.argv) > 1:

        # Running sudoku solver with one board $python3 sudoku.py <input_string>.
        start = time()
        print(sys.argv[1])
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = {ROW[r] + COL[c]: int(sys.argv[1][9*r+c])
                 for r in range(9) for c in range(9)}

        # print(board)
        print_board(board)

        solved_board = backtracking(board)
        end = time()
        times.append(end - start)

        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(board_to_string(solved_board))
        outfile.write('\n')

        print("Finished all boards in file.")

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

        # Solve each board using backtracking
        for line in sudoku_list.split("\n"):

            start = time()
            if len(line) < 9:
                continue

            # Parse boards to dict representation, scanning board L to R, Up to Down
            board = {ROW[r] + COL[c]: int(line[9*r+c])
                     for r in range(9) for c in range(9)}

            # Print starting board. TODO: Comment this out when timing runs.
            print_board(board)

            # Solve with backtracking
            solved_board = backtracking(board)
            end = time()
            times.append(end - start)

            # Print solved board. TODO: Comment this out when timing runs.
            print_board(solved_board)

            # Write board to file
            outfile.write(board_to_string(solved_board))
            outfile.write('\n')

        print("Finishing all boards in file.")

    # print(solved_board)
    write_to_file(times)
