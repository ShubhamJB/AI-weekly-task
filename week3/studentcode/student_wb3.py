
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        my_index = -1
        next_soln = self.open_list[my_index]
        self.open_list.pop(my_index)
        # <==== insert your pseudo-code and code above here
        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        my_index = 0
        next_soln = self.open_list[my_index]
        self.open_list.pop(my_index)
        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if not self.open_list:
            return None
        else:
            def GetMemberWithHighestQuality(open_list):
                min_index = 0
                for i in range(len(open_list)):
                    if open_list[i].quality < open_list[min_index].quality:
                        min_index = i
                return min_index
            best_index = GetMemberWithHighestQuality(self.open_list)
            next_soln = self.open_list[best_index]
            self.open_list.pop(best_index)
        # <==== insert your pseudo-code and code above here
        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star  search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if not self.open_list:
            return None
        else:
            def GetMemberWithHighestCombinedScore(open_list):
                min_index = 0
                min_value = len(open_list[0].variable_values) + open_list[0].quality

                for i in range(1, len(open_list)):
                    value = len(open_list[i].variable_values) + open_list[i].quality
                    if value < min_value:
                        min_value = value
                        min_index = i

                return min_index
            best_index = GetMemberWithHighestCombinedScore(self.open_list)
            next_soln = self.open_list.pop(best_index)
        # <==== insert your pseudo-code and code above here
        return next_soln
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work
    # base maze
    maze = Maze(mazefile="maze.txt")

    maze.contents[3][4] = hole_colour  # Open path at (3, 4) to lure DFS
    maze.contents[8][4] = wall_colour  # Block path at (8,4) with wall for DFS dead-end

    maze.contents[10][6] = hole_colour  # Open path at (10,6) to trap DFS
    maze.contents[14][6] = wall_colour  # Block path at (14,6) with wall for dead-end
    maze.contents[16][1] = hole_colour  # Open path at (16,1) for DFS dead-end
    maze.contents[19][4] = hole_colour  # Open path at (19,4) for DFS dead-end

    maze.contents[8][1] = hole_colour  # Open path at (8,1) to mislead DFS
    maze.contents[12][9] = wall_colour  # Block path at (12,9) with wall for DFS
    maze.contents[11][12] = wall_colour  # Block path at (11,12) with wall for DFS
    maze.contents[9][2] = wall_colour  # Block path at (9,2) with wall for DFS
    maze.contents[10][19] = wall_colour # Block path at (10,19) with wall for DFS
    maze.contents[18][5] = wall_colour   # Block path at (18,5) with wall for DFS

    # Save the maze
    maze.save_to_txt("maze-breaks-depth.txt")
    # <==== insert your code above here

def create_maze_depth_better():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work
    # base maze
    maze = Maze(mazefile="maze.txt")
    
    maze.contents[1][8] = wall_colour  # Block path at (1,8) with wall to force DFS to reroute
    maze.contents[9][10] = wall_colour  # Block path at (9,10) with wall to create DFS obstacle
    maze.contents[15][6] = wall_colour  # Block path at (15,6) with wall to hinder DFS progress
    maze.contents[13][2] = wall_colour  # Block path at (13,2) with wall to limit DFS options
    maze.contents[12][13] = wall_colour  # Block path at (12,13) with wall to block DFS path
    maze.contents[2][13]=wall_colour  # Block path at (2,13) with wall to create DFS barrier

    # Save the maze
    maze.save_to_txt("maze-depth-better.txt")
    # <==== insert your code above here
