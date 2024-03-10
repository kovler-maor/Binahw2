import itertools
from copy import deepcopy

ids = ["313598674", "312239296"]
RETRIEVE_TREASURE_REWARD = 4
RESET_ENVIRONMENT_PENALTY = -2
MARINE_PENALTY = -1


class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = initial
        self.V = {}
        # self.value_iteration(self.generate_all_valid_states())

    def all_pirate_ships_combos(self):
        game_map = self.initial["map"]
        possible_locations = []
        for i in range(len(game_map)):
            for j in range(len(game_map[i])):
                if game_map[i][j] != "I":
                    possible_locations.append((i, j))

        pirate_ships_combos_dict = {}
        for pirate_ship in self.initial["pirate_ships"]:
            pirate_ships_combos_dict[pirate_ship] = []
            for location in possible_locations:
                for capacity in range(0, self.initial["pirate_ships"][pirate_ship]["capacity"] + 1):
                    pirate_ships_combos_dict[pirate_ship].append((pirate_ship, location, capacity))
        # generate all possible combinations of pirate ships and their locations
        general_pirate_ships_combos = list(itertools.product(*pirate_ships_combos_dict.values()))
        return general_pirate_ships_combos

    def all_treasures_combos(self):
        treasures_combos_dict = {}
        for treasure in self.initial["treasures"]:
            treasures_combos_dict[treasure] = []
            for location in self.initial["treasures"][treasure]["possible_locations"]:
                treasures_combos_dict[treasure].append((treasure, location))
        # generate all possible combinations of treasures and their locations
        general_treasures_combos = list(itertools.product(*treasures_combos_dict.values()))
        return general_treasures_combos

    def all_marine_ships_combos(self):
        marine_ships_combos_dict = {}
        for marine_ship in self.initial["marine_ships"]:
            marine_ships_combos_dict[marine_ship] = []
            for index in range(len(self.initial["marine_ships"][marine_ship]["path"])):
                marine_ships_combos_dict[marine_ship].append((marine_ship, index))
        # generate all possible combinations of marine ships and their locations
        general_marine_ships_combos = list(itertools.product(*marine_ships_combos_dict.values()))
        return general_marine_ships_combos

    def generate_all_states(self):
        pirate_ships_combos = self.all_pirate_ships_combos()
        treasures_combos = self.all_treasures_combos()
        marine_ships_combos = self.all_marine_ships_combos()
        all_states = list(itertools.product(pirate_ships_combos, treasures_combos, marine_ships_combos))
        return all_states

    def new_representation(self, state):
        pirate_ships = state["pirate_ships"]
        treasures = state["treasures"]
        marine_ships = state["marine_ships"]

        new_state_pirate_ships = []
        new_state_treasures = []
        new_state_marine_ships = []

        for pirate_ship in pirate_ships:
            new_state_pirate_ships.append(
                (pirate_ship, pirate_ships[pirate_ship]["location"], pirate_ships[pirate_ship]["capacity"])
            )

        for treasure in treasures:
            new_state_treasures.append(
                (treasure, treasures[treasure]["location"])
            )

        for marine_ship in marine_ships:
            new_state_marine_ships.append(
                (marine_ship, marine_ships[marine_ship]["index"])
            )

        # make the new state a tuple of tuples
        new_state = (tuple(new_state_pirate_ships), tuple(new_state_treasures), tuple(new_state_marine_ships))
        return new_state

    def old_representation(self, state):
        """
        This method should return the old representation of the state given the new representation
        :param state: tuple of tuples
        :return: state: dict of dicts
        for example: (pirate_ship_1, (0, 0), 2), (treasure_1, (4, 4)), (marine_1, 1)
        and the init state is:
        {'pirate_ships':
                        {'pirate_ship_1':
                                        {"location": (0, 0), "capacity": 2}
                        },
        'treasures':
                        {'treasure_1':
                                        {"location": (4, 4),
                                         "possible_locations": ((4, 4),),
                                          "prob_change_location": 0.5}
                        },
        'marine_ships':
                        {'marine_1':
                                        {"index": 0, "path": [(2, 3), (2, 2)]}
                        }
        }
        should return:
                {'pirate_ships':
                        {'pirate_ship_1':
                                        {"location": (0, 0), "capacity": 2}
                        },
        'treasures':
                        {'treasure_1':
                                        {"location": (4, 4),
                                         "possible_locations": ((4, 4),),
                                          "prob_change_location": 0.5}
                        },
        'marine_ships':
                        {'marine_1':
                                        {"index": 1, "path": [(2, 3), (2, 2)]}
                        }
        }

        """
        pirate_ships = self.initial["pirate_ships"]
        treasures = self.initial["treasures"]
        marine_ships = self.initial["marine_ships"]

        new_state_pirate_ships = {}
        new_state_treasures = {}
        new_state_marine_ships = {}

        pirate_ships_new_state = state[0]
        treasures_new_state = state[1]
        marine_ships_new_state = state[2]

        for pirate_ship in pirate_ships:
            for i in range(len(pirate_ships_new_state)):
                if pirate_ship == pirate_ships_new_state[i][0]:
                    new_state_pirate_ships[pirate_ship] = {
                        "location": pirate_ships_new_state[i][1],
                        "capacity": pirate_ships_new_state[i][2]
                    }

        for treasure in treasures:
            for i in range(len(treasures_new_state)):
                if treasure == treasures_new_state[i][0]:
                    new_state_treasures[treasure] = {
                        "location": treasures_new_state[i][1],
                        "possible_locations": self.initial["treasures"][treasure]["possible_locations"],
                        "prob_change_location": self.initial["treasures"][treasure]["prob_change_location"]
                    }

        for marine_ship in marine_ships:
            for i in range(len(marine_ships_new_state)):
                if marine_ship == marine_ships_new_state[i][0]:
                    new_state_marine_ships[marine_ship] = {
                        "index": marine_ships_new_state[i][1],
                        "path": marine_ships[marine_ship]["path"]
                    }

        return {
            "map": self.initial["map"],
            "pirate_ships": new_state_pirate_ships,
            "treasures": new_state_treasures,
            "marine_ships": new_state_marine_ships
        }

        return new_state

    def all_possible_actions_for_state(self, state: dict):
        """
        A function that returns all possible actions for the state
        :param state:
        :return:
        """
        """
        A function that returns all possible actions of the state as a list of tuples
        :param state:
        :return: list of all possible combinations of the actions of the state for each state and each pirate ship
        and also add the "Terminate" action and the "reset" action
        for example:
        if we have 2 pirate ships and 1 treasure
        and the map is:
        ['S', 'S', 'I', 'S'],
        ['S', 'S', 'I', 'S'],
        ['B', 'S', 'S', 'S'],
        ['S', 'S', 'I', 'S']
        and the state is:
        {'optimal': True,
         'infinite': False,
         'map': [['S', 'S', 'I', 'S'], ['S', 'S', 'I', 'S'], ['B', 'S', 'S', 'S'], ['S', 'S', 'I', 'S']],
         'pirate_ships': {'pirate_ship_1': {"location": (0, 0), "capacity": 2},
                          'pirate_ship_2': {"location": (0, 1), "capacity": 3}},
         'treasures': {'treasure_1': {"location": (0, 2), "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                      "prob_change_location": 0.1}},
         'marine_ships': {'marine_1': {"index": 0, "path": [(1, 1), (2, 1)]}},
         'turns to go': 100}
        returns:
        [(('sail', 'pirate_ship_1', (0, 1)),('sail', 'pirate_ship_2', (0, 0)))-> this is one combination
        ....,(('Terminate',)),(('reset',))]

        """

        available_actions = []
        for pirate_ship in state["pirate_ships"]:
            ship_capacity = state["pirate_ships"][pirate_ship]["capacity"]
            ship_actions = []
            pirate_ship_position = state["pirate_ships"][pirate_ship]["location"]
            # Check if the pirate ship can SAIL to the next position
            # ------------------------------------------------------
            for action in [("sail", pirate_ship, (pirate_ship_position[0] + 1, pirate_ship_position[1])),
                           ("sail", pirate_ship, (pirate_ship_position[0] - 1, pirate_ship_position[1])),
                           ("sail", pirate_ship, (pirate_ship_position[0], pirate_ship_position[1] + 1)),
                           ("sail", pirate_ship, (pirate_ship_position[0], pirate_ship_position[1] - 1))]:
                if 0 <= action[2][0] < len(state["map"]) and 0 <= action[2][1] < len(state["map"][0]):
                    if state["map"][action[2][0]][action[2][1]] != "I":
                        ship_actions.append(action)

            # End of SAIL action
            # ------------------------------------------------------

            # Check if the pirate ship can COLLECT TREASURE from the current position
            # ------------------------------------------------------
            for treasure in state["treasures"]:
                treasure_position = state["treasures"][treasure]["location"]
                # check if the pirate ship is adjacent to the treasure
                if ((abs(pirate_ship_position[0] - treasure_position[0]) + abs(
                        pirate_ship_position[1] - treasure_position[1]) == 1)
                        and ship_capacity > 0):
                    ship_actions.append(("collect_treasure", pirate_ship, treasure))
            # End of COLLECT TREASURE action
            # ------------------------------------------------------

            # Check if the pirate ship can Deposit TREASURE from the current position
            # ------------------------------------------------------
            if state["map"][pirate_ship_position[0]][pirate_ship_position[1]] == "B":
                ship_actions.append(("deposit_treasure", pirate_ship))
            # End of DEPOSIT TREASURE action
            # ------------------------------------------------------

            # Add the WAIT action
            # ------------------------------------------------------
            ship_actions.append(("wait", pirate_ship))
            # End of WAIT action
            # ------------------------------------------------------
            available_actions.append(ship_actions)

        all_actions = list(itertools.product(*available_actions))
        all_actions.append((("Terminate",),))
        all_actions.append((("reset",),))
        return list(all_actions)

    def all_possible_next_states_from_action(self, state: dict, action: tuple):
        """
        A function that returns all possible next states from the action
        :param state: dict of the current state
        :param action: tuple of the action
        :return: list of all possible next states from the action
        """
        state = deepcopy(state)
        next_states = []
        pirate_ships = state["pirate_ships"]
        treasures = state["treasures"]
        marine_ships = state["marine_ships"]

        initial_state = {"pirate_ships": deepcopy(state["pirate_ships"]), "treasures": deepcopy(state["treasures"]),
                         "marine_ships": deepcopy(state["marine_ships"]), "map": deepcopy(state["map"])}

        for act in action:
            if act[0] == "reset":
                new_state = deepcopy(self.initial)
                next_states.append(initial_state)
            else:
                new_state = deepcopy(state)
                action_type = act[0]
                pirate_ship = act[1]
                pirate_ship_position = pirate_ships[pirate_ship]["location"]
                if action_type == "sail":
                    new_state = state.copy()
                    new_state["pirate_ships"][pirate_ship]["location"] = act[2]

                if action_type == "collect_treasure":
                    new_state = state.copy()
                    new_state["pirate_ships"][pirate_ship]["capacity"] -= 1

                if action_type == "deposit_treasure":
                    new_state = state.copy()
                    new_state["pirate_ships"][pirate_ship]["capacity"] = \
                        self.initial["pirate_ships"][pirate_ship]["capacity"]

                if action_type == "wait":
                    continue

        marines = self.initial["marine_ships"]
        all_possible_locations = []
        for marine in marines:
            path = marines[marine]["path"]
            all_possible_locations.append(list(range(0, len(path))))
        marines_locations = list(itertools.product(*all_possible_locations))

        treasures = self.initial["treasures"]
        all_possible_locations = []
        for treasure in treasures:
            possible_locations = treasures[treasure]["possible_locations"]
            all_possible_locations.append(possible_locations)
        treasures_locations = list(itertools.product(*all_possible_locations))

        for marines_locations in marines_locations:
            for treasures_location in treasures_locations:
                new_state = deepcopy(state)
                for place, marine in enumerate(marines):
                    new_state["marine_ships"][marine]["index"] = marines_locations[place]
                for place, treasure in enumerate(treasures):
                    new_state["treasures"][treasure]["location"] = treasures_location[place]
                next_states.append(new_state)

        return next_states

    def p_s_a_s_prime(self, state: dict, next_state: dict):
        """
        P(s' | s, a) = probability of transitioning to state s' assuming action a was taken in state s was valid
        :param state: dict of the current state
        :param next_state: dict of the next state
        :return: the probability of transitioning to state s' assuming action a was taken in state s was valid -> float
        """
        state_treasures = state["treasures"]
        next_state_treasures = next_state["treasures"]

        state_marine_ships = state["marine_ships"]
        next_state_marine_ships = next_state["marine_ships"]

        # The next part will calculate the probability of transitioning the
        # treasures from the current state to the next state
        # ----------------------------------------------------------------
        treasures_transition_probability = 1
        for treasure in state_treasures:
            number_of_available_locations = len(state_treasures[treasure]["possible_locations"])
            treasure_change_location_probability = (state_treasures[treasure]["prob_change_location"] *
                                                    1 / number_of_available_locations)
            treasure_static_location_probability = ((1 - state_treasures[treasure]["prob_change_location"]) +
                                                    (state_treasures[treasure]["prob_change_location"] *
                                                     1 / number_of_available_locations))

            if state_treasures[treasure]["location"] != next_state_treasures[treasure]["location"]:
                treasures_transition_probability *= treasure_change_location_probability
            else:
                treasures_transition_probability *= treasure_static_location_probability
        # ----------------------------------------------------------------

        # The next part will calculate the probability of transitioning the marine ships
        # from the current state to the next state
        # ----------------------------------------------------------------
        marine_ships_transition_probability = 1
        for marine in state_marine_ships:
            marine_index = state_marine_ships[marine]["index"]
            # marine can't really move
            if len(state_marine_ships[marine]["path"]) == 1:
                continue
            # check if the index is 0 or the last index
            if marine_index == 0 or marine_index == len(state_marine_ships[marine]["path"]) - 1:
                marine_ships_transition_probability *= 0.5
            else:
                marine_ships_transition_probability *= 1 / 3
        # ----------------------------------------------------------------
        return treasures_transition_probability * marine_ships_transition_probability

    def r_s_a(self, state: dict, action: tuple):
        """
        R(s,a)
       A function that returns the reward of the action in the state
       point given for each state as such:
           1. Successfully retrieving a treasure to base: 4 points.
           2. Resetting the environment: -2 points.
           3. Encountering a marine:
                -1 points for each ship that encounters a marine.
                For example, if 2 ships encounter marines, then you get -2 points.
                Applies for ships with and without treasures.
       :param dict state: dictionary of the state
       :param action: tuples of the action
       :return: the expected reward of the action in the state
        """

        # initializing the reward list
        reward = 0
        # getting the pirate ships, marine ships and treasures
        pirate_ships = state["pirate_ships"]
        marine_ships = state["marine_ships"]

        # The next part is a new version of the reward function where we would calculate the reward for
        # encountering a marine in this current state and from the next state
        # -----------------------------------------------------------------------------------------------
        pirate_ships_locations = [pirate_ships[ship]["location"] for ship in pirate_ships]

        marine_ships_locations = [marine_ships[marine]["path"]
                                  [marine_ships[marine]["index"]] for marine in marine_ships]

        for pirate_ship_location in pirate_ships_locations:
            for marine_location in marine_ships_locations:
                if pirate_ship_location == marine_location:
                    reward += MARINE_PENALTY
        # -----------------------------------------------------------------------------------------------

        # The next part is the old version of the reward function where we would calculate the reward for the
        # atomic actions. We would calculate the reward for each action and then sum them up.
        # (Sail and Wait don't have a reward or penalty so we are not calculating them)
        # -----------------------------------------------------------------------------------------------
        for act in action:
            # NOTE: action is a list of tuples with specific action for each ship ((...), (...))

            if act[0] == "reset":  # Tax for resetting the environment
                return RESET_ENVIRONMENT_PENALTY

            if act[0] == "Terminate":  # Tax for resetting the environment
                return 0

            action_type = act[0]
            pirate_ship = act[1]
            if action_type == "deposit_treasure":  # Reward for successfully retrieving a treasure to base
                maximum_capacity_of_ship = self.initial["pirate_ships"][pirate_ship]["capacity"]
                current_capacity_of_ship = pirate_ships[pirate_ship]["capacity"]
                number_of_treasures_on_ship = maximum_capacity_of_ship - current_capacity_of_ship
                reward += (RETRIEVE_TREASURE_REWARD * number_of_treasures_on_ship)

        return reward

    def init_V(self, states: list):
        """
        Initialize the value function V(s) to zero for all states s
        :param states: list of all valid states for each ship, treasure, and marine ship as tuples
        """
        V = {}
        for state in states:
            self.V[state, 0] = {"max_value": 0, "max_action": None}
        self.V = V

    def value_iteration(self, states: list):
        # TODO: implement this method
        """
        Implement the value iteration algorithm
        :return:
        """
        # initializing the value function V(s) to zero for all states s at time t = 0
        self.init_V(states)

        turns = 1
        while turns <= self.initial["turns to go"]:
            for state in states:
                max_value = float("-inf")
                max_action = None
                for action in self.all_possible_actions_for_state(state):
                    value = 0
                    # R(s, a)
                    reward = self.r_s_a(state, action)
                    # sum(P(s'|s, a) * V(s'))
                    sum_of_transition_probabilities = 0
                    for next_state in self.all_possible_next_states_from_action(state, action):
                        # P(s'|s, a) * V(s')
                        sum_of_transition_probabilities += (
                                self.p_s_a_s_prime(state, next_state) * self.V[next_state, turns - 1]["max_value"])

                    if value >= max_value:
                        max_value = value
                        max_action = action
                self.V[state, turns] = {"max_value": max_value, "max_action": max_action}
            turns += 1

    def act(self, state):
        raise NotImplemented


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
