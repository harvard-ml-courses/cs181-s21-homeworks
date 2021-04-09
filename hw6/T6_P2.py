import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gridworld

# ----------------------------------------------------------------------- #
#   Starter code for CS 181 2020 HW 6, Problem 2                          #
# ----------------------------------------------------------------------- #
#
# Please read all of T6_P2.py before beginning to code.

#############################################################
#          DO NOT MODIFY THIS REGION OF THE CODE.           #

np.random.seed(181)
VALUE_ITER = 'Value'
POLICY_ITER = 'Policy'

### HELPER CODE ####

### INIITALIZE GRID ###

# Create the grid for Problem 2.
grid = [
    '..,..',
    '..,..',
    'o.,..',
    '.?,.*']

# Create the Task
# Task Parameters

task = gridworld.GridWorld(grid,
                            terminal_markers={'*', '?'},
                            rewards={'.': -1, '*': 50, '?': 5, ',': -50, 'o': -1} )

# Algorithm Parameters
gamma = .75
state_count = task.num_states
action_count = task.num_actions
row_count = len(grid)
col_count = len(grid[0])

# -------------- #
#   Make Plots   #
# -------------- #

# Util to make an arrow
# The directions are [ 'north' , 'south' , 'east' , 'west' ]
def plot_arrow( location , direction , plot ):
    arrow = plt.arrow( location[0] , location[1] , dx , dy , fc="k", ec="k", head_width=0.05, head_length=0.1 )
    plot.add_patch(arrow)

# Util to draw the value function V as numbers on a plot.
def make_value_plot(V):
    # Useful stats for the plot
    value_function = np.reshape( V , ( row_count , col_count ) )

    # Write the value on top of each square
    indx, indy = np.arange(row_count), np.arange(col_count)

    fig, ax = plt.subplots()
    ax.imshow( value_function , interpolation='none' , cmap= plt.get_cmap('Reds_r'))

    s = 0
    for s in range(len(V)):
        val = V[s]
        (xval, yval) = task.maze.unflatten_index(s)
        t = "%.2f"%(val,) # format value with 1 decimal point

        ax.text(yval, xval, t, color='black', va='center', ha='center', size=15)


# Util to draw the policy pi as arrows on a plot.
def make_policy_plot(pi, iter_type = VALUE_ITER, iter_num = 0):
    # Useful stats for the plot
    row_count = len( grid )
    col_count = len( grid[0] )
    policy_function = np.reshape( pi , ( row_count , col_count ) )

    for row in range( row_count ):
        for col in range( col_count ):
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow( col , row , dx , dy , shape='full', fc='w' , ec='gray' , lw=1., length_includes_head=True, head_width=.1 )
    plt.title( iter_type + ' Iteration, i = ' + str(iter_num) )
    plt.savefig(iter_type + '_' + str(iter_num) + '.png')
    plt.show(block=True)
    # Save the plot to the local directory.

#############################################################
#          HELPER FUNCTIONS - DO NOT MODIFY                 #

def unflatten_index(flat_i):
    """
    flat_i is an flattened state index.
    Returns the unflattened representation of the state at flat_i.
    """
    return task.maze.unflatten_index(flat_i)

def flatten_index(unflat_i):
    """
    unflat_i is an unflattened state index.
    Returns the flattened representation of the state at unflat_i.
    """
    return task.maze.flatten_index(unflat_i)

def get_reward(state, action):
    """
    state is a flattened state.
    action represents an index into the actions array.
    Returns the reward from taking an action (deterministically, with
    no probability of failure) from state.
    """

    new_state = task.move(task.maze.unflatten_index(state), action)
    if task.is_wall(new_state):
        # If the (state, action) pair is a wall, return the reward
        # at the present state.
        return task.rewards.get(task.maze.get_flat(state))
    return task.rewards.get(task.maze.get_flat(state), action)

def is_wall(state):
    """
    state represents a flattened state.
    Returns true if state is a wall.
    """
    if state < 0 or state > state_count:
        return True
    return task.is_wall(task.maze.unflatten_index(state))

def get_transition_prob(state1, action1, state2):
    """
    state1 and state2 are flattened states.
    action1 represents an index into the actions array.
    Returns p(state2 | state1, action1).
    """
    return task.get_transition_prob(state1, action1, state2)

def move(state, action):
    """
    state is a flattened state.
    action represents an index into the actions array.
    Returns the flattened index of the state resulting from moving
    in direction action from state.
    Returns -1 if the action is hitting a wall.
    """
    (x, y) = task.move(task.maze.unflatten_index(state), action)
    if x < 0 or y < 0 or x >= row_count or y >= col_count:
        return -1
    else:
        return task.maze.flatten_index((x, y))

#############################################################
#          TO-DOS FOR PROBLEM 2                             #

def use_helper_functions():
    """
    This function contains several examples of how to use the helper functions
    in this class.
    """

    # In Gridworld, each state on the grid can be represented using an
    # unflattened or a flattened index.
    # The unflattened index is a tuple (x, y) representing the state's position
    # on the grid.
    # The flattened index is a single integer. Each integer in range(state_count)
    # corresponds to a particular state.
    # Here is some code to convert unflattened indices to flattened indices.

    # Unflattened indices -> flattened indices.
    for r_ind in range(row_count):
        for c_ind in range(col_count):
            u_ind = (r_ind, c_ind)
            print(str(u_ind) + ' is converted to ' + str(flatten_index(u_ind)))

    # Flattened indices -> unflattened indices.
    for s in range(state_count):
        # Convert flattened index s to an unflattened index s_u.
        s_u = unflatten_index(s)
        print(str(s) + ' is converted to ' + str(s_u))

    # Recall that when you take an action in Gridworld, you won't always
    # necessarily move in that direction.  Instead there is some probability of
    # moving to a state on either side.

    starting_state = 14
    for a in range(action_count):
        for new_state in range(state_count):
            # Only print when new_state has nonzero probability
            # p(new state | starting_state, a).
            prb = get_transition_prob(starting_state, a, new_state)
            if prb > 0:
                print('Going from ' + str(unflatten_index(starting_state)) + ' to '
                    + str(unflatten_index(new_state)) + ' when taking action '
                    + str(task.actions[a]) + ' has probability ' + str(prb))



def policy_evaluation(pi, gamma, theta = 0.1):
    """
    Returns matrix V containing the policy evaluation of policy pi.
    Implement policy evaluation using discount factor gamma and
    convergence tolerance theta.
    
    In your implementation, please use helper functions get_transition_prob
    and get_reward in this file.  You do not need to calculate the transition
    probabilities yourself.
    """
    # TODO: Complete this function.
    V = np.zeros(state_count)

    return V

def update_policy_iteration(V, pi, gamma, theta = 0.1):
    """
    Return updated V_new and pi_new using policy iteration.
    V represents the learned value function at each state.
    pi represents the learned policy at each state.

    In your implementation, please use helper functions get_transition_prob
    and get_reward in this file.  You do not need to calculate the transition
    probabilities yourself.

    Implement only one iteration of policy iteration in this function, which
    is called multiple times in the for loop in the learn_strategy function
    at the end of the file.
    """
    # TODO: Complete this function.
    V_new = policy_evaluation(pi, gamma, theta)
    pi_new = np.zeros(state_count)

    return V_new, pi_new

def update_value_iteration(V, pi, gamma):
    """
    Update V and pi using value iteration.
    V represents the learned value function at each state.
    pi represents the learned policy at each state.

    In your implementation, please use helper functions get_transition_prob
    and get_reward in this file.  You do not need to calculate the transition
    probabilities yourself.

    Implement only one iteration of value iteration in this function, which
    is called multiple times in the for loop in the learn_strategy function
    at the end of the file.
    """
    # TODO: Complete this function.
    V_new = np.zeros(state_count)
    pi_new = np.zeros(state_count)

    return V_new, pi_new

"""
Do not modify the learn_strategy method, but read through its code.
Parameters
----------
planning_type: string
    Specifies whether value or policy iteration is used to learn the strategy.
max_iter: int
    The maximum number of iterations (i.e. number of updates) the learning
    policy will be run for.
print_every: int
    The frequency at which the function will save value and policy plots.
ct: float
    The convergence tolerance used for policy or value iteration.
"""
def learn_strategy(planning_type = VALUE_ITER, max_iter = 10, print_every = 5, ct = None):
    # Loop over some number of episodes
    V = np.zeros(state_count)
    pi = np.zeros(state_count)

    # Update Q-table using value/policy iteration until max iterations or until ct reached
    for n_iter in range(max_iter):
        V_prev = V.copy()

        # Update V and pi using value iteration.
        if planning_type == VALUE_ITER:
            V, pi = update_value_iteration(V, pi, gamma)

            if (n_iter % print_every == 0):
                # make value plot
                make_value_plot(V = V)
                # plot the policy
                make_policy_plot(pi = pi, iter_type = VALUE_ITER, iter_num = n_iter)

            # Calculate the difference between this V and the previous V.
            diff = np.absolute(np.subtract(V, V_prev))

            # Check that every state's difference is less than the convergence tol.
            if ct and np.max(diff) < ct:
                print("Converged at iteration " + str(n_iter))
                # make value plot
                make_value_plot(V = V)
                # plot the policy
                make_policy_plot(pi = pi, iter_type = VALUE_ITER, iter_num = n_iter)
                return 0

        # Update V and pi using policy iteration.
        elif planning_type == POLICY_ITER:
            V, pi = update_policy_iteration(V, pi, gamma, theta = ct)
            if (n_iter % print_every == 0):
                # make value plot
                make_value_plot(V = V)
                # plot the policy
                make_policy_plot(pi = pi, iter_type = POLICY_ITER, iter_num = n_iter)


print('Beginning policy iteration.')
learn_strategy(planning_type=POLICY_ITER, max_iter = 10, print_every = 2)
print('Policy iteration complete.')

print('Beginning value iteration.')
learn_strategy(planning_type=VALUE_ITER, max_iter = 10, print_every = 2)
print('Value iteration complete.\n')
