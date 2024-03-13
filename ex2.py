import itertools
from copy import deepcopy

ids = ["313598674", "312239296"]
RETRIEVE_TREASURE_REWARD = 4
RESET_ENVIRONMENT_PENALTY = -2
MARINE_PENALTY = -1


class OptimalPirateAgent:
    """
    This class should implement the OptimalPirateAgent which results the best value function for the given state
    """

    def __init__(self, initial: dict) -> None:
        """
        This initiate the agent with the given state
        :param initial: dict of the problem initial state
        """
        self.initial = None  # the initial state of the agent
        self.V = None  # the value function of the agent
        self.turn = None  # the number of turns to go the game has
        self.set_initial_state(initial)  # set the initial state of the agent
        self.value_iteration(self.generate_all_states())  # calculate the value function of the agent

    def set_initial_state(self, initial: dict) -> None:
        """
        this function should set the initial state of the agent
        :param initial: dict of the problem initial state
        :return: nothing
        """
        self.initial = initial
        self.V = {}
        self.turn = deepcopy(self.initial["turns to go"])

    def all_pirate_ships_combos(self) -> list:
        """
        This function should return all possible combinations of pirate ships and their locations
        :return: list of all possible combinations of pirate ships and their locations
        """
        game_map = self.initial["map"]  # the game map
        possible_locations = []  # all possible locations for the pirate ships
        for i in range(len(game_map)):  # iterate over the game map
            for j in range(len(game_map[i])):
                if game_map[i][j] != "I":  # if the cell is not an island then it is a possible location
                    possible_locations.append((i, j))

        pirate_ships_combos_dict = {}  # a dictionary to store the possible locations for each pirate ship
        for pirate_ship in self.initial["pirate_ships"]:  # iterate over the pirate ships
            pirate_ships_combos_dict[pirate_ship] = []  # initialize the list of possible locations for the pirate ship
            for location in possible_locations:  # iterate over the possible locations
                for capacity in range(0, self.initial["pirate_ships"][pirate_ship]["capacity"] + 1):
                    pirate_ships_combos_dict[pirate_ship].append((pirate_ship, location, capacity))
        # generate all possible combinations of pirate ships and their locations
        general_pirate_ships_combos = list(itertools.product(*pirate_ships_combos_dict.values()))
        return general_pirate_ships_combos

    def all_treasures_combos(self) -> list:
        """
        This function should return all possible combinations of treasures and their locations
        :return: list of all possible combinations of treasures and their locations
        """
        treasures_combos_dict = {}  # a dictionary to store the possible locations for each treasure
        for treasure in self.initial["treasures"]:  # iterate over the treasures
            treasures_combos_dict[treasure] = []  # initialize the list of possible locations for the treasure
            for location in self.initial["treasures"][treasure]["possible_locations"]:
                treasures_combos_dict[treasure].append((treasure, location))  # add the location to the list
        # generate all possible combinations of treasures and their locations
        general_treasures_combos = list(itertools.product(*treasures_combos_dict.values()))
        return general_treasures_combos

    def all_marine_ships_combos(self) -> list:
        """
        This function should return all possible combinations of marine ships and their locations
        :return:
        """
        marine_ships_combos_dict = {}  # a dictionary to store the possible locations for each marine ship
        for marine_ship in self.initial["marine_ships"]:  # iterate over the marine ships
            marine_ships_combos_dict[marine_ship] = []  # initialize the list of possible locations for the marine ship
            for index in range(len(self.initial["marine_ships"][marine_ship]["path"])):
                marine_ships_combos_dict[marine_ship].append((marine_ship, index))  # add the location to the list
        # generate all possible combinations of marine ships and their locations
        general_marine_ships_combos = list(itertools.product(*marine_ships_combos_dict.values()))
        return general_marine_ships_combos

    def generate_all_states(self) -> list:
        """
        This function should return all possible states of the game
        :return: list of all possible states of the game
        """
        # generate all possible combinations of pirate ships and their locations
        pirate_ships_combos = self.all_pirate_ships_combos()

        # generate all possible combinations of treasures and their locations
        treasures_combos = self.all_treasures_combos()

        # generate all possible combinations of marine ships and their locations
        marine_ships_combos = self.all_marine_ships_combos()

        # generate all possible states of the game
        all_states = list(itertools.product(pirate_ships_combos, treasures_combos, marine_ships_combos))
        return all_states

    def new_representation(self, state: dict) -> tuple:
        """
        This method should return the new representation of the state given the old representation
        which looks like this: (pirate_ship_1, (0, 0), 2), (treasure_1, (4, 4)), (marine_1, 1)
        given this: {'pirate_ships': {'pirate_ship_1': {"location": (0, 0), "capacity": 2}},
                        'treasures': {'treasure_1': {"location": (4, 4),
                                                    "possible_locations": ((4, 4),),
                                                    "prob_change_location": 0.5}},
                        'marine_ships': {'marine_1': {"index": 0, "path": [(2, 3), (2, 2)]}}
                        }
        :param state:
        :return: tuple of tuples
        """
        pirate_ships = state["pirate_ships"]  # the pirate ships
        treasures = state["treasures"]  # the treasures
        marine_ships = state["marine_ships"]  # the marine ships

        new_state_pirate_ships = []  # the new representation of the pirate ships
        new_state_treasures = []  # the new representation of the treasures
        new_state_marine_ships = []  # the new representation of the marine ships

        for pirate_ship in pirate_ships:  # iterate over the pirate ships
            new_state_pirate_ships.append(
                (pirate_ship, pirate_ships[pirate_ship]["location"], pirate_ships[pirate_ship]["capacity"])
            )

        for treasure in treasures:  # iterate over the treasures
            new_state_treasures.append(
                (treasure, treasures[treasure]["location"])
            )

        for marine_ship in marine_ships:  # iterate over the marine ships
            new_state_marine_ships.append(
                (marine_ship, marine_ships[marine_ship]["index"])
            )

        # make the new state a tuple of tuples
        new_state = (tuple(new_state_pirate_ships), tuple(new_state_treasures), tuple(new_state_marine_ships))
        return new_state

    def old_representation(self, state: tuple) -> dict:
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
        pirate_ships = self.initial["pirate_ships"]  # the pirate ships
        treasures = self.initial["treasures"]  # the treasures
        marine_ships = self.initial["marine_ships"]  # the marine ships

        new_state_pirate_ships = {}  # the new representation of the pirate ships
        new_state_treasures = {}  # the new representation of the treasures
        new_state_marine_ships = {}  # the new representation of the marine ships

        pirate_ships_new_state = state[0]  # the new representation of the pirate ships
        treasures_new_state = state[1]  # the new representation of the treasures
        marine_ships_new_state = state[2]  # the new representation of the marine ships

        for pirate_ship in pirate_ships:  # iterate over the pirate ships
            for i in range(len(pirate_ships_new_state)):  # iterate over the new representation of the pirate ships
                if pirate_ship == pirate_ships_new_state[i][0]:
                    new_state_pirate_ships[pirate_ship] = {
                        "location": pirate_ships_new_state[i][1],
                        "capacity": pirate_ships_new_state[i][2]
                    }

        for treasure in treasures:  # iterate over the treasures
            for i in range(len(treasures_new_state)):  # iterate over the new representation of the treasures
                if treasure == treasures_new_state[i][0]:  # if the treasure is the same as
                    # the new representation of the treasure
                    new_state_treasures[treasure] = {
                        "location": treasures_new_state[i][1],
                        "possible_locations": self.initial["treasures"][treasure]["possible_locations"],
                        "prob_change_location": self.initial["treasures"][treasure]["prob_change_location"]
                    }

        for marine_ship in marine_ships:  # iterate over the marine ships
            for i in range(len(marine_ships_new_state)):  # iterate over the new representation of the marine ships
                if marine_ship == marine_ships_new_state[i][0]:  # if the marine ship is the same as
                    new_state_marine_ships[marine_ship] = {
                        "index": marine_ships_new_state[i][1],
                        "path": marine_ships[marine_ship]["path"]
                    }
        # make the new state a tuple of tuples
        return {
            "map": self.initial["map"],
            "pirate_ships": new_state_pirate_ships,
            "treasures": new_state_treasures,
            "marine_ships": new_state_marine_ships
        }

    def all_possible_actions_for_state(self, state: dict) -> list:
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
        ....,'terminate','reset']

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
                    ship_actions.append(("collect", pirate_ship, treasure))
            # End of COLLECT TREASURE action
            # ------------------------------------------------------

            # Check if the pirate ship can Deposit TREASURE from the current position
            # ------------------------------------------------------
            if state["map"][pirate_ship_position[0]][pirate_ship_position[1]] == "B":
                ship_actions.append(("deposit", pirate_ship))
            # End of DEPOSIT TREASURE action
            # ------------------------------------------------------

            # Add the WAIT action
            # ------------------------------------------------------
            ship_actions.append(("wait", pirate_ship))
            # End of WAIT action
            # ------------------------------------------------------
            available_actions.append(ship_actions)
        all_actions = ["terminate", "reset"]
        all_actions.extend(itertools.product(*available_actions))
        return list(all_actions)

    def all_possible_next_states_from_action(self, state: dict, action: tuple) -> list:
        """
        A function that returns all possible next states from the action
        :param state: dict of the current state
        :param action: tuple of the action
        :return: list of all possible next states from the action in their old representation
        """
        state = deepcopy(state)
        next_states = []
        pirate_ships = state["pirate_ships"]
        treasures = state["treasures"]
        marine_ships = state["marine_ships"]

        initial_state = {"pirate_ships": deepcopy(state["pirate_ships"]), "treasures": deepcopy(state["treasures"]),
                         "marine_ships": deepcopy(state["marine_ships"]), "map": deepcopy(state["map"])}
        if action == "reset":
            new_state = deepcopy(self.initial)
            next_states.append(initial_state)
        elif action == "terminate":
            pass
        else:
            new_state = deepcopy(state)
            for act in action:
                action_type = act[0]
                pirate_ship = act[1]
                pirate_ship_position = pirate_ships[pirate_ship]["location"]
                if action_type == "sail":
                    new_state = state.copy()
                    new_state["pirate_ships"][pirate_ship]["location"] = act[2]

                if action_type == "collect":
                    new_state = state.copy()
                    new_state["pirate_ships"][pirate_ship]["capacity"] -= 1

                if action_type == "deposit":
                    new_state = state.copy()
                    new_state["pirate_ships"][pirate_ship]["capacity"] = \
                        self.initial["pirate_ships"][pirate_ship]["capacity"]

                if action_type == "wait":
                    pass

            # Checking encounters with marines
            marine_ships_locations = \
                [marine_ships[marine]["path"][marine_ships[marine]["index"]] for marine in marine_ships]
            for pirate_ship in pirate_ships:
                pirate_ship_location = pirate_ships[pirate_ship]["location"]
                if pirate_ship_location in marine_ships_locations:
                    initial_capacity = self.initial["pirate_ships"][pirate_ship]["capacity"]
                    new_state["pirate_ships"][pirate_ship]["capacity"] = initial_capacity

            marines = new_state["marine_ships"]
            all_possible_locations = []
            for marine in marines:
                path = marines[marine]["path"]
                index = marines[marine]["index"]
                if len(path) == 1:
                    all_possible_locations.append([index])
                else:
                    if index == 0:
                        all_possible_locations.append([index, index + 1])
                    elif index == len(path) - 1:
                        all_possible_locations.append([index, index - 1])
                    else:
                        all_possible_locations.append([index - 1, index, index + 1])
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

    def p_s_a_s_prime(self, state: dict, next_state: dict) -> float:
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
        p = treasures_transition_probability * marine_ships_transition_probability
        return p

    def r_s_a(self, state: dict, action: tuple) -> float:
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

        # NOTE: action is a list of tuples with specific action for each ship ((...), (...))

        if action == "reset":  # Tax for resetting the environment
            return RESET_ENVIRONMENT_PENALTY

        if action == "terminate":  # Tax for resetting the environment
            return 0
        for act in action:
            action_type = act[0]
            pirate_ship = act[1]
            if action_type == "deposit":  # Reward for successfully retrieving a treasure to base
                maximum_capacity_of_ship = self.initial["pirate_ships"][pirate_ship]["capacity"]
                current_capacity_of_ship = pirate_ships[pirate_ship]["capacity"]
                number_of_treasures_on_ship = maximum_capacity_of_ship - current_capacity_of_ship
                reward += (RETRIEVE_TREASURE_REWARD * number_of_treasures_on_ship)

        return reward

    def init_V(self, states: list) -> None:
        """
        Initialize the value function V(s) to zero for all states s
        :param states: list of all valid states for each ship, treasure, and marine ship as tuples
        """
        V = {}
        for state in states:
            V[state, 0] = {"max_value": 0, "max_action": None}
        self.V = V

    def value_iteration(self, states: list) -> None:
        """
        Implement the value iteration algorithm
        :return: nothing
        """
        # initializing the value function V(s) to zero for all states s at time t = 0
        self.init_V(states)

        turns = 1
        while turns <= self.initial["turns to go"]:
            for state in states:
                state = self.old_representation(state)
                max_value = float("-inf")
                max_action = None
                A = self.all_possible_actions_for_state(state)
                for action in A:
                    value = 0
                    # R(s, a)
                    reward = self.r_s_a(state, action)
                    # sum(P(s'|s, a) * V(s'))
                    if action[0][0] == "reset":
                        value = (RESET_ENVIRONMENT_PENALTY +
                                 self.V[self.new_representation(self.initial), turns - 1]["max_value"])
                        if value >= max_value:
                            max_value = value
                            max_action = action
                    else:
                        sum_of_transition_probabilities = 0
                        for next_state in self.all_possible_next_states_from_action(state, action):
                            # P(s'|s, a) * V(s')
                            sum_of_transition_probabilities += \
                                (
                                        self.p_s_a_s_prime(state, next_state) *
                                        self.V[self.new_representation(next_state), turns - 1]["max_value"]
                                )
                        value = reward + sum_of_transition_probabilities

                        if value >= max_value:
                            max_value = value
                            max_action = action

                state = self.new_representation(state)
                self.V[state, turns] = {"max_value": max_value, "max_action": max_action}

            turns += 1

    def act(self, state):
        turn = state["turns to go"]
        state = self.new_representation(state)
        act = self.V[state, turn]["max_action"]
        print(f"-------------------{turn}-------------------")
        print(f"Pirate ship: {self.old_representation(state)['pirate_ships']}")
        print(f"State: {self.old_representation(state)}")
        print(f"Act: {act}")

        if act[0][0] == "deposit":
            print(f"Deposited")
        if self.old_representation(state)["pirate_ships"]["pirate_ship_1"]["location"] == (2, 3):
            print(f"Encountered a marine")

        return act


class PirateAgent(OptimalPirateAgent):
    """
    This class should implement the PirateAgent which results the best value function for the given state
    after reducing the state space by using the following strategies:
    1. Choose the best treasure to go to based on the distance from the base
    2. Create a mini map of the game map
    3. Get the relevant marine ships
    4. Get the relevant pirate ships
    5. Get the relevant treasures
    6. Reduce the state space by using the above strategies
    7. Implement the value iteration algorithm
    """

    def __init__(self, initial: dict) -> None:
        """
        This initiate the agent with the given state after reducing the state space
        :param initial:
        """
        super().__init__(initial)
        self.original_state = deepcopy(initial)  # save the original state
        self.set_initial_state(initial)  # set the initial state of the agent after reducing the state space
        self.V = {}  # the value function of the agent
        self.turn = deepcopy(self.initial["turns to go"])  # the number of turns to go the game has
        self.value_iteration(self.generate_all_states())  # calculate the value function of the agent

    def set_initial_state(self, initial: dict) -> None:
        """
        this function should set the initial state of the agent after reducing the state space
        using the strategies mentioned in the class docstring.
        :param initial: dict of the problem initial state
        :return: nothing
        """

        def choose_best_treasure() -> dict:
            """
            This function should choose the best treasure to go to based on the distance from the base
            This is the first strategy
            :return: dict of the best treasure
            """

            stats_of_treasures = {}
            base_position = initial["pirate_ships"]["pirate_ship_1"]["location"]
            # all of the marines possible locations
            marine_ships = initial["marine_ships"]
            marine_ships_locations = set()
            for marine in marine_ships:
                marine_path = marine_ships[marine]["path"]
                for location in marine_path:
                    marine_ships_locations.add(location)

            for treasure in initial["treasures"]:
                treasure_stats = initial["treasures"][treasure]
                treasure_position = treasure_stats["location"]
                distance = abs(base_position[0] - treasure_position[0]) + abs(base_position[1] - treasure_position[1])
                stats_of_treasures[treasure] = distance
                stats_of_treasures[treasure] = {"initial_distance": distance,
                                                "prob_change_location": treasure_stats["prob_change_location"],
                                                "initial_location": treasure_position}

            shortest_distance = float("inf")
            best_treasure = None
            for treasure in stats_of_treasures:
                initial_distance = stats_of_treasures[treasure]["initial_distance"]
                if initial_distance < shortest_distance:
                    shortest_distance = initial_distance
                    best_treasure = {treasure: initial["treasures"][treasure]}
            return best_treasure

        def create_mini_map() -> dict:
            """
            This function should create a mini map of the game map
            :return: dict of the mini map and the maximum and minimum x and y
            """
            treasure = choose_best_treasure()  # the best treasure to go to
            first_treasure = list(treasure.keys())[0]  # the first treasure
            possible_locations = treasure[first_treasure]["possible_locations"]
            # getting the base position
            base_position = initial["pirate_ships"]["pirate_ship_1"]["location"]
            all_possible_locations = set()
            for location in possible_locations:
                all_possible_locations.add(location)

            all_possible_locations.add(base_position)

            # getting the maximum and minimum x and y
            max_x = max([location[0] for location in all_possible_locations])
            min_x = min([location[0] for location in all_possible_locations])
            max_y = max([location[1] for location in all_possible_locations])
            min_y = min([location[1] for location in all_possible_locations])

            # creating the mini map
            mini_map = []
            for i in range(min_x, max_x + 1):
                row = []
                for j in range(min_y, max_y + 1):
                    cell = initial["map"][i][j]
                    row.append(cell)
                mini_map.append(row)
            return {"mini_map": mini_map, "max_x": max_x, "min_x": min_x, "max_y": max_y, "min_y": min_y}

        def get_relevant_marine_ships() -> dict:
            """
            This function should return the relevant marine ships
            :return: dict of the relevant marine ships
            """
            marine_ships = initial["marine_ships"]
            mini_map = create_mini_map()
            all_positions_in_mini_map = []
            relevant_marine_ships = {}
            for i in range(mini_map["min_x"], mini_map["max_x"] + 1):
                for j in range(mini_map["min_y"], mini_map["max_y"] + 1):
                    all_positions_in_mini_map.append((i, j))

            for marine_ship in marine_ships:
                marine_path = marine_ships[marine_ship]["path"]
                for position in marine_path:
                    if position in all_positions_in_mini_map:
                        relevant_marine_ships[marine_ship] = marine_ships[marine_ship]
                        break
            return relevant_marine_ships

        pirate_ships = initial["pirate_ships"]  # the pirate ships
        treasures = choose_best_treasure()  # the best treasure to go to
        mini_map = create_mini_map()["mini_map"]  # the mini map
        marine_ships = get_relevant_marine_ships()  # the relevant marine ships

        self.initial = {
            "optimal": True,
            "infinite": False,
            "map": mini_map,
            "pirate_ships": pirate_ships,
            "treasures": treasures,
            "marine_ships": marine_ships,
            "turns to go": 100
        }

    def all_pirate_ships_combos(self) -> list:
        """
        This function should return all possible combinations of pirate ships and their locations
        it overrides the parent class method to reduce the state space by making the pirate ships
        do the same actions.
        :return:
        """
        game_map = self.initial["map"]  # the game map
        possible_locations = []  # the possible locations
        for i in range(len(game_map)):
            for j in range(len(game_map[i])):
                if game_map[i][j] != "I":
                    possible_locations.append((i, j))

        minimum_capacity = float("inf")
        for pirate_ship in self.initial["pirate_ships"]:
            if self.initial["pirate_ships"][pirate_ship]["capacity"] < minimum_capacity:
                minimum_capacity = self.initial["pirate_ships"][pirate_ship]["capacity"]

        general_pirate_ships_combos = []
        for capacity in range(0, minimum_capacity + 1):
            for location in possible_locations:
                current_combination = []
                for pirate_ship in self.initial["pirate_ships"]:
                    current_combination.append((pirate_ship, location, capacity))
                general_pirate_ships_combos.append(tuple(current_combination))

        return general_pirate_ships_combos

    def all_possible_actions_for_state(self, state: dict) -> list:
        """
        A function that returns all possible actions of the state as a list of tuples
        it overrides the parent class method to reduce the state space by making the pirate ships
        do the same actions.
        :param state:
        :return:
        """
        available_actions = []

        minimum_capacity = float("inf")
        for pirate_ship in state["pirate_ships"]:
            if state["pirate_ships"][pirate_ship]["capacity"] < minimum_capacity:
                minimum_capacity = state["pirate_ships"][pirate_ship]["capacity"]

        first_pirate_ship = list(state["pirate_ships"].keys())[0]
        pirate_ships_names = list(state["pirate_ships"].keys())
        pirate_ship_position = state["pirate_ships"][first_pirate_ship]["location"]
        # Sail action
        for action in [("sail", first_pirate_ship, (pirate_ship_position[0] + 1, pirate_ship_position[1])),
                       ("sail", first_pirate_ship, (pirate_ship_position[0] - 1, pirate_ship_position[1])),
                       ("sail", first_pirate_ship, (pirate_ship_position[0], pirate_ship_position[1] + 1)),
                       ("sail", first_pirate_ship, (pirate_ship_position[0], pirate_ship_position[1] - 1))]:
            if 0 <= action[2][0] < len(state["map"]) and 0 <= action[2][1] < len(state["map"][0]):
                if state["map"][action[2][0]][action[2][1]] != "I":
                    current_combination = []
                    for pirate_ship in pirate_ships_names:
                        current_combination.append((action[0], pirate_ship, action[2]))
                    available_actions.append(tuple(current_combination))

        # Collect action
        for treasure in state["treasures"]:
            treasure_position = state["treasures"][treasure]["location"]
            if ((abs(pirate_ship_position[0] - treasure_position[0]) + abs(
                    pirate_ship_position[1] - treasure_position[1]) == 1)
                    and minimum_capacity > 0):
                current_combination = []
                for pirate_ship in pirate_ships_names:
                    current_combination.append(("collect", pirate_ship, treasure))
                available_actions.append(tuple(current_combination))

        # Deposit action
        if state["map"][pirate_ship_position[0]][pirate_ship_position[1]] == "B":
            current_combination = []
            for pirate_ship in pirate_ships_names:
                current_combination.append(("deposit", pirate_ship))
            available_actions.append(tuple(current_combination))

        # Wait action
        current_combination = []
        for pirate_ship in pirate_ships_names:
            current_combination.append(("wait", pirate_ship))
        available_actions.append(tuple(current_combination))

        all_actions = ["terminate", "reset"]
        all_actions.extend(available_actions)
        return list(all_actions)

    def act(self, state: dict) -> tuple:
        """
        This function should return the best action for the given state
        it overrides the parent class because the state space is reduced,
        and we need to make sure that the agent is acting based on the reduced state space
        :param state:
        :return:
        """
        turn = state["turns to go"]

        def state_transformation(state: dict) -> dict:
            """
            This function should transform the state to the old representation
            which matches the whole state space from the initial state
            :param state: dict
            :return: state: dict
            """
            treasures = state["treasures"]
            marine_ships = state["marine_ships"]

            relevant_marine_ships = self.initial["marine_ships"].keys()
            relevant_treasures = self.initial["treasures"].keys()

            new_marine_ships = {}
            new_treasures = {}

            for marine_ship in marine_ships:
                if marine_ship in relevant_marine_ships:
                    new_marine_ships[marine_ship] = marine_ships[marine_ship]

            for treasure in treasures:
                if treasure in relevant_treasures:
                    new_treasures[treasure] = treasures[treasure]

            state["map"] = self.initial["map"]
            state["treasures"] = new_treasures
            state["marine_ships"] = new_marine_ships
            return state

        state = state_transformation(state)
        state = self.new_representation(state)
        act = self.V[state, turn]["max_action"]
        print(f"-------------------{turn}-------------------")
        print(f"Pirate ship: {self.old_representation(state)['pirate_ships']}")
        print(f"State: {self.old_representation(state)}")
        print(f"Act: {act}")

        if act[0][0] == "deposit":
            print(f"Deposited")

        return act


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
