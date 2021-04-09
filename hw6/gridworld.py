# DO NOT MODIFY THIS FILE.
# This file represents util methods to represent the grid.
# You do not need to call or modify any of these methods to complete Problem 2.

import numpy as np

# Maze state is represented as a 2-element NumPy array: (Y, X). Increasing Y is South.

# Possible actions, expressed as (delta-y, delta-x).
maze_actions = {
    'N': np.array([-1, 0]),
    'S': np.array([1, 0]),
    'E': np.array([0, 1]),
    'W': np.array([0, -1]),
}

def parse_topology(topology):
    return np.array([list(row) for row in topology])


class Maze(object):
    """
    Simple wrapper around a NumPy 2D array to handle flattened indexing and staying in bounds.
    """
    def __init__(self, topology):
        self.topology = parse_topology(topology)
        self.flat_topology = self.topology.ravel()
        self.shape = self.topology.shape

    def in_bounds_flat(self, position):
        return 0 <= position < np.product(self.shape)

    def in_bounds_unflat(self, position):
        return 0 <= position[0] < self.shape[0] and 0 <= position[1] < self.shape[1]

    def get_flat(self, position):
        if not self.in_bounds_flat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.flat_topology[position]

    def get_unflat(self, position):
        return self.topology[tuple(position)]

    def flatten_index(self, index_tuple):
        return np.ravel_multi_index(index_tuple, self.shape)

    def unflatten_index(self, flattened_index):
        return np.unravel_index(flattened_index, self.shape)

    def flat_positions_containing(self, x):
        return list(np.nonzero(self.flat_topology == x)[0])

    def flat_positions_not_containing(self, x):
        return list(np.nonzero(self.flat_topology != x)[0])

    def __str__(self):
        return '\n'.join(''.join(row) for row in self.topology.tolist())

    def __repr__(self):
        return 'Maze({})'.format(repr(self.topology.tolist()))


class GridWorld(object):
    """

    Parameters
    ----------

    maze : list of strings or lists
        maze topology (see below)

    rewards: dict of int (representing state on grid) to number.
        Rewards obtained by being in a maze grid with the specified contents,
        or experiencing the specified event (either 'hit-wall' or 'moved'). The
        contributions of content reward and event reward are summed. For
        example, you might specify a cost for moving by passing
        rewards={, 'moved': -1}.

    terminal_markers: sequence of chars, default '*'
        A grid cell containing any of these markers will be considered a
        "terminal" state.

    Notes
    -----

    Maze topology is expressed textually. Key:
     '#': wall
     '*': goal
     'o': origin
    """

    def __init__(self, maze, rewards={'*': 10}, terminal_markers='*', directions="NSEW"):

        self.maze = Maze(maze) if not isinstance(maze, Maze) else maze
        self.rewards = rewards
        self.terminal_markers = terminal_markers

        self.actions = [maze_actions[direction] for direction in directions]
        self.num_actions = len(self.actions)
        self.state = None
        self.num_states = self.maze.shape[0] * self.maze.shape[1]

    def __repr__(self):
        return 'GridWorld(maze={maze!r}, rewards={rewards}, terminal_markers={terminal_markers}})'.format(**self.__dict__)

    def is_wall(self, pos):
        """
        Returns true if pos is a wall.

        pos is an unflattened position in the grid.
        """
        (x, y) = pos
        return (x < 0 or x > 3 or y < 0 or y > 4)

    def move(self, state, action):
        """
        Returns a tuple containing the new position of moving in the
        direction of the action from the state.

        state is an unflattened position in the grid.
        action is the index into the actions[] array.
        """
        return tuple((state + self.actions[action]).reshape(1, -1)[0])

    def get_side_states(self, action, state):
        """
        Returns an array of the "side states" when taking action beginning
        at position state.

        Does not return states which are walls.

        Takes as input an unflattened start state.
        """

        side_states = []
        if action == 0 or action == 1:

            if not self.is_wall(self.move(state, 3)):
                side_states.append(self.move(state, 3))
            if not self.is_wall(self.move(state, 2)):
                side_states.append(self.move(state, 2))

        elif action == 2 or action == 3:
            if not self.is_wall(self.move(state, 0)):
                side_states.append(self.move(state, 0))
            if not self.is_wall(self.move(state, 1)):
                side_states.append(self.move(state, 1))

        return side_states

    def get_transition_prob(self, s1, action1, s2):
        """
        Takes as input flattened states s1, s2.
        Returns p(s2 | s1, action1).
        """
        state1 = self.maze.unflatten_index(s1)
        state2 = self.maze.unflatten_index(s2)

        new_state = self.move(state1, action1)

        sstates = self.get_side_states(action1, state1)
        succeed_prb = 0.8
        slip_prb = 0.1

        # One of the side states was a wall: adjust probabilities accordingly.
        if len(sstates) == 1:
            succeed_prb = 0.9

        if self.is_wall(new_state):
            if(state1 == state2):
                return succeed_prb
        else:
            if(state2 == new_state):
                return succeed_prb

        # Oherwise, check if state2 is on either side
        for side_state in sstates:
            if(state2 == side_state):
                return slip_prb

        return 0.


