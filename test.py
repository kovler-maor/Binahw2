from ex2 import InfinitePirateAgent, OptimalPirateAgent, PirateAgent, ids
from inputs import small_inputs

state1 = small_inputs[0]
state2 = small_inputs[1]


# TESTING THE ACTION FUNCTION
# ----------------------------------------------
agent = OptimalPirateAgent(state1)

actions1 = agent.actions(state1)

for action in actions1:
    print(f"action: {action}")
    print(f"expected reward: {agent.reward(state1, action)}")

# ----------------------------------------------
