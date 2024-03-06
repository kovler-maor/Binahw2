




def transition_function(self, state, action):
    """Returns the list of state that results from executing the given
    action in the given state. The action must be one of
    self.actions(state)."""
    states_and_probabilities = {}
    new_state = deepcopy(state)
    for act in action:
        pirate_ship = act[1]
        action_type = act[0]
        # SAIL action
        if action_type == "sail":
            new_state["pirate_ships"][pirate_ship]["location"] = act[2]
        # End of SAIL action
        # COLLECT TREASURE action
        elif action_type == "collect_treasure":
            treasure = act[2]
            new_state["pirate_ships"][pirate_ship]["capacity"] -= 1
        # End of COLLECT TREASURE action
        # DEPOSIT TREASURE action
        elif action_type == "deposit_treasure":
            new_state["pirate_ships"][pirate_ship]["capacity"] = self.ships[pirate_ship]["capacity"]
        # End of DEPOSIT TREASURE action
        # WAIT action
        elif action_type == "wait":
            pass
        # End of WAIT action

        initial_state = deepcopy(new_state)

        for treasure in new_state["treasures"]:

            moving_treasure_prob = (new_state["treasures"][treasure]["prob_change_location"] *
                                    (1 / (len(new_state["treasures"][treasure]["possible_locations"]))))

            treasure_initial_location = initial_state["treasures"][treasure]["location"]

            for location in new_state["treasures"][treasure]["possible_locations"]:
                if location == treasure_initial_location:
                    treasure_prob = (1 - new_state["treasures"][treasure]["prob_change_location"] +
                                     moving_treasure_prob)
                else:
                    treasure_prob = moving_treasure_prob
                new_state["treasures"][treasure]["location"] = location

                for marine in new_state["marine_ships"]:
                    marine_stats = new_state["marine_ships"][marine]
                    index = marine_stats["index"]
                    size_of_path = len(marine_stats["path"])
                    new_state["turns to go"] -= 1
                    # If the marine is in the first index of the path
                    if index == 0:
                        marine_prob = 0.5
                        for i in [0, 1]:
                            new_state["marine_ships"][marine]["index"] = i
                            str_new_state = str(new_state)
                            states_and_probabilities[str_new_state] = marine_prob * treasure_prob
                    # If the marine is in the last index of the path
                    elif index == size_of_path - 1:
                        marine_prob = 0.5
                        for i in [index, index - 1]:
                            new_state["marine_ships"][marine]["index"] = i
                            str_new_state = str(new_state)
                            states_and_probabilities[str_new_state] = marine_prob * treasure_prob
                    # If the marine is in the middle of the path
                    else:
                        marine_prob = 1 / 3
                        for i in [index - 1, index, index + 1]:
                            new_state["marine_ships"][marine]["index"] = i
                            str_new_state = str(new_state)
                            states_and_probabilities[str_new_state] = marine_prob * treasure_prob

    return states_and_probabilities
def state_all_possible_outcomes(self, state):
    """
    A function that returns all possible outcomes of the state
    :param state: dictionary of the current state
    :return: list of all possible outcomes of the state (list of dictionaries)
    """
    all_possible_outcomes = {}
    for action in self.actions(state):
        out_comes_from_action = self.transition_function(state, action)
        all_possible_outcomes.update(out_comes_from_action)
    return all_possible_outcomes
