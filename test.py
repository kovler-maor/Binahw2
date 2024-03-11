from ex2 import InfinitePirateAgent, OptimalPirateAgent, PirateAgent, ids
from inputs_mine import small_inputs

state1 = small_inputs[0]

# ---------------------init test-------------------------
agent = OptimalPirateAgent(state1)

# ---------------------generate_all_state test-------------------------
all_possible_states = agent.generate_all_states()

# ---------------------old_representation test-------------------------
first_state = all_possible_states[1]
old_state = agent.old_representation(first_state)

# ---------------------new_representation test-------------------------
new_state = agent.new_representation(old_state)

for state in all_possible_states:
    old_representation = agent.old_representation(state)
    new_representation = agent.new_representation(old_representation)
    if state != new_representation:
        print("Error in new_representation")
        break
print("new_representation test passed")

# ---------------------all_possible_actions_for_state test-------------------------
all_actions = agent.all_possible_actions_for_state(old_state)

# ---------------------all_possible_next_states_from_action test-------------------------
action = all_actions[4]
all_next_states = agent.all_possible_next_states_from_action(old_state, action)

# print(all_next_states)

# ---------------------p_s_a_s_prime test-------------------------
# next_state = all_next_states[0]
# p = agent.p_s_a_s_prime(old_state, next_state)

all_probs = {}
for index, next_state in enumerate(all_next_states):
    p = agent.p_s_a_s_prime(old_state, next_state)
    if p < 0 or p > 1:
        print("Error in p_s_a_s_prime")
        break
    else:
        all_probs[index] = p
print("sum of all probs: ", sum(all_probs.values()))
if sum(all_probs.values()) < 0.99:
    print("Error in p_s_a_s_prime")
else:
    print("p_s_a_s_prime test passed")
print(sum(all_probs.values()))

# ---------------------r_s_a test-------------------------
r_a = agent.r_s_a(old_state, action)
print(r_a)
# ---------------------value_iteration test-------------------------
agent.value_iteration(all_possible_states)
print("value_iteration test passed")
