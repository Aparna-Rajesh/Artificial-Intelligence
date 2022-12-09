from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
import resource
from tracemalloc import start
from typing import List, Tuple

import heapq

#### SKELETON CODE ####


# The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """

    def __init__(self, config, n, parent=None, action="Initial", cost=0, priority=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception(
                "Config contains invalid/duplicate entries : ", config)

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.config = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

        # I'm adding the priority to this for a*
        self.priority = priority

    def __lt__(self, other):  # added this for the pq that we use in a*
        return self.priority < other.priority

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            # TODO: self.n used to be 3
            print(self.config[3*i: 3*(i+1)])
            # print(self.config[self.n*i: self.n*(i+1)])

    def move_up(self):
        """
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """

        """
        3 4 5               1 2 3  4
        0 1 2               0 5 6  7
        6 7 8               8 9 10 11
        """
        if self.blank_index < self.n:  # if it's already in the top row!
            return None
        else:
            # store the index of the value of the piece above the blank
            new_config = list(self.config)

            new_config[self.blank_index], new_config[self.blank_index -
                                                     self.n] = new_config[self.blank_index - self.n], new_config[self.blank_index]

            new_state = PuzzleState(
                new_config, self.n, self, "Up", self.cost + 1)
            # new_state.blank_index = new_index #dont actually need this because it will calculate it when we create a PuzzleState

            return new_state

    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index > ((self.n * (self.n - 1)) - 1):  # if it's already in the bottom row!
            return None
        else:
            new_config = list(self.config)

            new_config[self.blank_index], new_config[self.blank_index +
                                                     self.n] = new_config[self.blank_index + self.n], new_config[self.blank_index]

            new_state = PuzzleState(
                new_config, self.n, self, "Down", self.cost + 1)

            return new_state

    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index % self.n == 0:
            return None
        else:
            new_config = list(self.config)
            new_config[self.blank_index], new_config[self.blank_index -
                                                     1] = new_config[self.blank_index - 1], new_config[self.blank_index]

            new_state = PuzzleState(
                new_config, self.n, self, "Left", self.cost + 1)

            return new_state

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index % self.n == self.n - 1:
            return None
        else:
            new_config = list(self.config)
            new_config[self.blank_index], new_config[self.blank_index +
                                                     1] = new_config[self.blank_index + 1], new_config[self.blank_index]

            new_state = PuzzleState(
                new_config, self.n, self, "Right", self.cost + 1)

            return new_state

    def expand(self):
        """ Generate the child nodes of this node """

        # Node has already been expanded
        if len(self.children) != 0:
            return self.children

            # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children


class Frontier(object):
    """
    one major bottleneck of list, dequeue, or queue class in Python is that their membership
    testing operation is O(n). The membership testing speed is critical for search algorithm
    because that operation is executed for every child state. Coming up with using such
    list-like data structures is a good first step, but for using it with reasonable execution
    time, you might need one more trick.
    """

    def __init__(self):
        self.frontier_as_a_q = Q.Queue()

        self.frontier_as_a_list = []

        # sets use hashing, so it will be easier to check membership if we use a set in tandem with the q/lst
        self.frontier_as_a_set = set()

        self.frontier_as_a_pq = Q.PriorityQueue()

    def in_set(self, state) -> Tuple:
        """
        Checking whether the given state's config was stored inside the set or not
        """
        as_tuple = tuple(state.config)
        return as_tuple in self.frontier_as_a_set

    # this is for bfs because bfs uses queues

    def add_to_q_and_set(self, state: PuzzleState):
        """
        Adding a state to both the queue and a set.
        When adding to the set, we only care about the config because it needs to be
        saved as a tuple in order to be hashable.
        https://rollbar.com/blog/handling-unhashable-type-list-exceptions/#:~:text=The%20Python%20TypeError%3A%20unhashable%20type,which%20is%20an%20unhashable%20object
        """
        self.frontier_as_a_q.put(state)
        self.frontier_as_a_set.add(tuple(state.config))

    def get_from_q_rem_from_set(self) -> PuzzleState:
        # get removes the first item in the queue (FIFO)
        state = self.frontier_as_a_q.get()
        self.frontier_as_a_set.remove(tuple(state.config))
        return state

    # this is for dfs, because we use lists in dfs:

    def add_to_list_and_set(self, state: PuzzleState):
        self.frontier_as_a_list.append(state)
        self.frontier_as_a_set.add(tuple(state.config))

    def pop_from_list_rem_from_set(self) -> PuzzleState:
        # pop removes the last element of the list (LIFO)
        state = self.frontier_as_a_list.pop()
        self.frontier_as_a_set.remove(tuple(state.config))
        return state

    # this is for a* because a* uses a priority queue and a set

    def add_to_pq_and_set(self, cost, state: PuzzleState):
        self.frontier_as_a_pq.put((cost, state))
        self.frontier_as_a_set.add(tuple(state.config))

    def rem_from_pq_rem_from_set(self) -> PuzzleState:
        state = self.frontier_as_a_pq.get()[1]
        self.frontier_as_a_set.remove(tuple(state.config))
        return state

    # Function that Writes to output.txt

    # Students need to change the method to have the corresponding parameters


def build_path(final_state: PuzzleState):
    path = []
    curr = final_state
    while curr.parent != None:
        # building path backwards, so place action in the front
        path.insert(0, curr.action)
        curr = curr.parent

    return path


def writeOutput(path: List, time_taken, ram_used, num_expanded, max_depth):
    # print(path)
    # print("time", format(time_taken, '.8f'))
    # print("ram", format(ram_used, '.8f'))
    # print("num_expanded", num_expanded)
    # print("search_depth", len(path))
    # print("max_depth", max_depth)

    with open("output.txt", "w") as w:
        w.write("path_to_goal: {}\n".format(path))
        w.write("cost_of_path: {}\n".format(len(path)))
        w.write("nodes_expanded: {}\n".format(num_expanded))
        w.write("search_depth: {}\n".format(len(path)))
        w.write("max_search_depth: {}\n".format(max_depth))
        w.write(f"running_time: {time_taken:.8f}\n")
        w.write(f"max_ram_usage: {ram_used:.8f}")


def bfs_search(initial_state):
    """
    BFS search

    function BREADTH-FIRST-SEARCH(problem) returns a solution node or failure
        node ← NODE(problem.INITIAL)
        if problem.IS-GOAL(node.STATE) then return node
        frontier ← a FIFO queue, with node as an element
        reached←{problem.INITIAL}
        while not IS-EMPTY(frontier)
            do node←POP(frontier)
            for each child in EXPAND(problem, node) do
                s←child.STATE
                if problem.IS-GOAL(s)
                    then return child
                if s is not in reached then
                    add s to reached
                    add child to frontier return failure
    """
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    bfs_start_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    frontier = Frontier()
    # add the initial state to both the q and the set

    frontier.add_to_q_and_set(initial_state)

    explored = set()
    num_expanded = 0
    depth = 0

    while not frontier.frontier_as_a_q.empty():

        # get the next item from frontier list and remove it from the set:
        curr = frontier.get_from_q_rem_from_set()
        explored.add(tuple(curr.config))

        # print(curr.config)
        if test_goal(curr):  # if we've reached the goal
            path = build_path(curr)
            end_time = time.time()  # stop the timer!
            dfs_end_ram = (resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss - bfs_start_ram) / (2**20)
            writeOutput(path, end_time - start_time,
                        dfs_end_ram, num_expanded, depth)
        else:
            list_of_children = curr.expand()
            num_expanded = num_expanded + 1
            for child in list_of_children:
                if not frontier.in_set(child) and tuple(child.config) not in explored:
                    # put the next non None child in the frontier set and queue
                    frontier.add_to_q_and_set(child)
                    if child.cost > depth:
                        depth = child.cost


def dfs_search(initial_state):
    """DFS search
        very similar to BFS but with a list now
        and the order that we remove from the list
        is modified to preserve the way that we
        populate the frontier
    """
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    bfs_start_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    frontier = Frontier()
    # add the initial state to both the list and the set
    frontier.add_to_list_and_set(initial_state)

    explored = set()
    num_expanded = 0
    depth = 0

    while len(frontier.frontier_as_a_list) > 0:

        curr = frontier.pop_from_list_rem_from_set()
        explored.add(tuple(curr.config))
        if test_goal(curr):
            path = path = build_path(curr)
            end_time = time.time()  # stop the timer!
            dfs_end_ram = (resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss - bfs_start_ram) / (2**20)
            writeOutput(path, end_time - start_time,
                        dfs_end_ram, num_expanded, depth)
        else:
            # in order to get it into the list correctly, we need to reverse
            list_of_children = curr.expand()
            list_of_children.reverse()
            num_expanded = num_expanded + 1
            for child in list_of_children:
                if not frontier.in_set(child) and tuple(child.config) not in explored:
                    frontier.add_to_list_and_set(child)
                    if child.cost > depth:
                        depth = child.cost


# WHY DOES THIS NOTTTTTT WORK?? It does work; I just can't read.
def A_star_search(initial_state):
    """A * search"""
    start_time = time.time()
    bfs_start_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    node_num = 0
    counter = 0
    frontier = Frontier()
    frontier.add_to_pq_and_set(0, initial_state)

    explored = set()
    depth = 0
    while not frontier.frontier_as_a_pq.empty():
        state = frontier.frontier_as_a_pq.get()[1]
        if tuple(state.config) in frontier.frontier_as_a_set:
            frontier.frontier_as_a_set.remove(tuple(state.config))
            explored.add(tuple(state.config))
            # state = frontier.rem_from_pq_rem_from_set()
            if test_goal(state):
                end_time = time.time()
                dfs_end_ram = (resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss - bfs_start_ram) / (2**20)
                path = build_path(state)
                # path: List, time_taken, ram_used, num_expanded, max_depth
                return writeOutput(path, end_time - start_time, bfs_start_ram - dfs_end_ram, counter, depth)
            else:
                children = state.expand()
                counter = counter + 1
                for child in children:
                    node_num = node_num + 1
                    child.priority = node_num
                    if tuple(child.config) not in explored:
                        distance = calculate_total_cost(child)
                        frontier.add_to_pq_and_set(distance, child)

                        if child.cost > depth:
                            depth = child.cost
        # else:
        #     # print(cost)
        #     print(str(cost) + " " + str(tuple(state.config)))


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    curr_value = state.cost

    # the total cost for a* is always the cost to reach node n + cost to get from n to the goal
    for i in range(0, state.n * state.n):  # for every index in the state
        if not state.config[i] == 0:  # skipping the 0 tile
            curr_value = curr_value + \
                (calculate_manhattan_dist(i, state.config[i], state.n))
    return curr_value


def calculate_manhattan_dist(idx, value, n):
    """
    Calculates the Manhattan distance of a tile.
    :param idx: index of start tile
    :param value: goal tile
    :param n: # of rows, n = 3 for our cases
    """
    # getting the distance it's off by for each tile besides the 0
    return abs(value % n - idx % n) + abs(value // n - idx // n)


def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    return puzzle_state.config == list(range(puzzle_state.n * puzzle_state.n))

# Main Function that reads in Input and Runs corresponding Algorithm


def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, board_size)
    start_time = time.time()

    if search_mode == "bfs":
        bfs_search(hard_state)
    elif search_mode == "dfs":
        dfs_search(hard_state)
    elif search_mode == "ast":
        A_star_search(hard_state)
    # elif search_mode == "ab":
        # hard_state.display()
        # print(calculate_total_cost(hard_state))
        #    print(calculate_manhattan_dist(0, hard_state.config[0], 3))
        # elif search_mode == "display":  # display is me
        #     if hard_state != None:
        #         hard_state.display()
        #     # print(hard_state.blank_index)
        #     # print("n ", hard_state.n)
        #     next_state = hard_state.move_down()
        #     if next_state != None:
        #         print()
        #         next_state.display()

        #     next_state = next_state.move_left()
        #     if next_state != None:
        #         print()
        #         next_state.display()

        #     next_state = next_state.move_down()
        #     if next_state != None:
        #         print()
        #         next_state.display()

        #     next_state = next_state.move_left()
        #     if next_state != None:
        #         print()
        #         next_state.display()

        #     next_state = next_state.move_left()
        #     if next_state != None:
        #         print()
        #         next_state.display()
    else:
        print("Enter valid command arguments !")

    end_time = time.time()
    print("Program completed in %.3f second(s)" % (end_time-start_time))


if __name__ == '__main__':
    main()
