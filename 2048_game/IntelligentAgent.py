
import random
from BaseAI import BaseAI
import math
import time

'''
You must use adversarial search in your IntelligentAgent (expectiminimax with alpha-beta pruning).
You must provide your move within the time limit of 0.2 seconds.
You must name your file IntelligentAgent.py (Python 3).

- Employ the expectiminimax algorithm. This is a requirement.
- Implement alpha-beta pruning. This is a requirement.
- Use heuristic functions.
- Assign heuristic weights. You will likely want to include more than one heuristic function.
'''


class IntelligentAgent(BaseAI):

    time_limit = None

    def if_end(self, depth):
        if depth > 10 or time.process_time() >= self.time_limit:
            return True
        False

    def getMove(self, grid):

        self.time_limit = time.process_time() + .2
        action = (self.maxmin_abpruning(grid, -math.inf, math.inf, 0))[0]
        if action == None:  # max recur depth reached or too much time
            random.choice(grid.getAvailableMoves())[1]
        return action

    def maxmin_abpruning(self, grid, alpha, beta, depth):

        optimal_move = None  # direction

        def pos_of_computer_action(grid, alpha, beta, depth):

            if self.if_end(depth):  # timeout
                return (None, self.set_of_heuristrics(grid))

            return 0.9 * self.computer_actions(grid, 2, alpha, beta, (1+depth)) + \
                0.1 * self.computer_actions(grid, 4,
                                            alpha, beta, (1+depth))

        if self.if_end(depth):
            return (None, self.set_of_heuristrics(grid))

        curr_max = -math.inf

        for pos_move in grid.getAvailableMoves():
            x = pos_of_computer_action(
                pos_move[1], alpha, beta, (depth+1))
            cur_util = x
            if type(x) is tuple:
                cur_util = x[1]

            if cur_util > curr_max:
                curr_max = cur_util
                optimal_move = pos_move[0]
            elif curr_max >= beta:  # prune
                break

            alpha = max(alpha, curr_max)

        return optimal_move, curr_max

    def computer_actions(self, grid, node_value, alpha, beta, depth):

        if self.if_end(depth):
            return self.set_of_heuristrics(grid)

        min_of_br = math.inf

        for move in grid.getAvailableCells():

            t_grid = grid.clone()
            t_grid.insertTile(move, node_value)

            curr_util = self.maxmin_abpruning(
                t_grid, alpha, beta, (depth + 1))
            if (curr_util[1] < min_of_br):
                min_of_br = curr_util[1]

            if min_of_br <= beta:  # or time limit
                break
            elif min_of_br < beta:
                beta = min_of_br

        return min_of_br

    def set_of_heuristrics(self, grid):

        def free_tiles():
            '''
            counting the number of free tiles
            in the game because  options can 
            quickly run out when the game board 
            gets too cramped
            '''
            free = 0
            for row in range(len(grid.map)):
                for col in range(len(grid.map[row])):
                    if grid.map[row][col] == 0:
                        free += 1
            return free

        def manhattan_distance():
            '''
            higher valued tiles should be clustered in a corner; 
            using top left corner because the numbers are much 
            nicer that way
            '''
            count = 0
            for row in range(len(grid.map)):
                for col in range(len(grid.map[row])):
                    dist = row + col
                    power = (len(grid.map) - 1) * 2 - dist
                    count += (2 ** power) * grid.map[row][col]
            return count

        def monotonicity():
            '''
            tries to ensure that the values of the 
            tiles are all either increasing or 
            decreasing along both the left/right 
            and up/down directions
            '''
            mono = 0
            for row in range(len(grid.map) - 1):
                for col in range(len(grid.map[0]) - 1):
                    if grid.map[row][col] >= grid.map[row][col+1] and grid.map[col][row] >= grid.map[col][row+1]:
                        mono += 1
                    if grid.map[row][col] >= grid.map[row][col+1] or grid.map[col][row] >= grid.map[col][row+1]:
                        mono += 1

            return mono

        score = free_tiles() + manhattan_distance() + monotonicity()
        return score
