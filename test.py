from inputs import small_inputs
from ex2 import InfinitePirateAgent, PirateAgent, ids, OptimalPirateAgent

first_input = small_inputs[2]

# TESTING THE ACTION FUNCTION
# ----------------------------------------------
agent = OptimalPirateAgent(first_input)

# The agent should be able to move the pirate ship to the right
# and collect the treasure
action = (('sail', 'pirate_ship_1', (0, 1)), ('sail', 'pirate_ship_2', (0, 1)))
test = agent.generate_all_possible_outcomes(first_input, action)

print(action)

# ----------------------------------------------
