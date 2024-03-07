import itertools
import random as rnd
from copy import deepcopy

ids = ["313598674", "312239296"]

RESET_PENALTY = 50
MARINE_PENALTY = -1
DEPOSIT_REWARD = 100

class OptimalPirateAgent:
    def __init__(self, initial):
        our_state = self.build_state(initial)
        self.initial = our_state
        self.all_possible_states = self.generate_all_possible_states()
        self.V = {}

    # FUNCTIONS FOR CREATING ALL THE POSSIBLE STATES
    # ----------------------------------------------
    def build_state(self, initial):
        """
        A function that builds the state from the given initial state
        :param initial:
        :return:
        """
        state = dict()
        state["optimal"] = initial["optimal"]
        state["infinite"] = initial["infinite"]
        state["map"] = initial["map"]
        state["pirate_ships"] = initial["pirate_ships"]
        state["treasures"] = initial["treasures"]
        state["marine_ships"] = initial["marine_ships"]
        state["turns to go"] = 100
        return state

    def all_possible_pirates_location(self):
        """
        A function that returns all possible locations of the pirate ships
        :return: list of all possible locations of the pirate ships
        for example:
        [['S', 'S', 'I', 'S'],
         ['S', 'S', 'I', 'S'],
         ['B', 'S', 'S', 'S'],
         ['S', 'S', 'I', 'S']]
        returns:
        all possible combinations of the locations of the pirate ships
        for example:
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),
         (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
         this is for one pirate ship
        partial example for two pirate ships:
        output:
        [((0, 0), (0, 0)), ((0, 0), (0, 1)), ((0, 0), (0, 2)), ((0, 0), (0, 3)), ((0, 0), (1, 1)), ((0, 0), (1, 3)...]
        which is all possible combinations of the locations of the pirate ships
        """
        state = self.initial
        game_map = state["map"]
        all_possible_locations = []
        for i in range(len(game_map)):
            for j in range(len(game_map[0])):
                if game_map[i][j] != "I":
                    all_possible_locations.append((i, j))

        list_for_combinations = [all_possible_locations for i in range(len(state["pirate_ships"]))]
        return list(itertools.product(*list_for_combinations))

    def all_possible_treasures_location(self):
        """
        A function that returns all possible locations of the treasures
        :return: list of all possible locations of the treasures
        for example:
        'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5},
        'treasure_2': {"location": (4, 4),
                                     "possible_locations": ((4, 4), (2, 2)),
                                     "prob_change_location": 0.5}
        returns:
        [((4,4),(4,4)),((4,4),(2,2))] each entry in each tuple is the location of the treasure according to
         their order in the initial state
        """
        treasures = self.initial["treasures"]
        all_possible_locations = []
        for treasure in treasures:
            possible_locations = treasures[treasure]["possible_locations"]
            all_possible_locations.append(possible_locations)
        return list(itertools.product(*all_possible_locations))

    def all_possible_marines_location(self):
        """
        A function that returns all possible locations of the marine ships (indexes)
        :return: list of all possible locations of the marine ships
        for example:
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the marine": {"index": 0,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                            }
        which means the marine_1 can be in index 0 or 1
        and larry the marine can be in index 0, 1 or 2
        returns:
        [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
        each entry in each tuple is the index of the marine ship according to their order in the initial state
        """
        marines = self.initial["marine_ships"]
        all_possible_locations = []
        for marine in marines:
            path = marines[marine]["path"]
            all_possible_locations.append(list(range(0, len(path))))
        return list(itertools.product(*all_possible_locations))

    def all_possible_capacities(self):
        """
        A function that returns all possible capacities of the pirate ships
        :return: list of all possible capacities of the pirate ships
        for example:
        for one ship with capacity 2
        returns:
        [(0,),(1,),(2,)]
        for two ships with capacities 2 and 3
        returns:
        [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]
        """
        pirate_ships = self.initial["pirate_ships"]
        capacities = []
        for ship in pirate_ships:
            capacity = pirate_ships[ship]["capacity"]
            capacities.append(list(range(0, capacity + 1)))
        return list(itertools.product(*capacities))

    def all_turns_to_go(self):
        """
        A function that returns all possible turns to go
        :return: list of all possible turns to go
        """
        number_of_turns = self.initial["turns to go"]
        return list(range(0, number_of_turns + 1))

    def generate_pirate_ships_dict(self, pirate_ships_location, capacities):
        """
        A function that generates a dictionary of the pirate ships
        :param pirate_ships_location:
        :param capacities:
        :return: dictionary of the pirate ships
        for example:
        if we have 2 priate ships and
        pirate_ships_location = [(0, 0), (0, 1)]
        and capacities = (2, 3)
        then the output will be:
        {'pirate_ship_1': {"location": (0, 0),
                           "capacity": 2},
         'pirate_ship_2': {"location": (0, 1),
                           "capacity": 3}
        }
        """
        pirate_ships_names = list(self.initial["pirate_ships"].keys())
        pirate_ships_dict = {}
        for i in range(len(pirate_ships_names)):
            pirate_ships_dict[pirate_ships_names[i]] = {"location": pirate_ships_location[i], "capacity": capacities[i]}
        return pirate_ships_dict

    def generate_marine_ships_dict(self, marine_ships_location):
        """
        A function that generates a dictionary of the marine ships
        :param marine_ships_location: list of the marine ships locations indexes
        :return: dict of the marine ships
        for example:
        if we have 2 marine ships and
        marine_ships_location = [(0, 0), (0, 1)]
        then the output will be:
        {
        'marine_1': {"index": 0, "path": [(1, 1), (2, 1)]--> the path is not changing}
         'marine_2': {"index": 1, "path": [(0, 1), (0, 2)]--> the path is not changing}
        }
        """
        marine_ships_names = list(self.initial["marine_ships"].keys())
        marine_ships_dict = {}

        for i in range(len(marine_ships_names)):
            marine_ship_path = self.initial["marine_ships"][marine_ships_names[i]]["path"]
            marine_ships_dict[marine_ships_names[i]] = {
                "index": marine_ships_location[i],
                "path": marine_ship_path
            }

        return marine_ships_dict

    def generate_treasures_dict(self, treasures_location):
        """
        A function that generates a dictionary of the treasures
        :param treasures_location:
        :return: dictionary of the treasures
        for example:
        if we have 2 treasures and
        treasures_location = [(4,4), (2, 2)]
        then the output will be:
        {'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5},
        'treasure_2': {"location": (4, 4),
                         "possible_locations": ((4, 4), (2, 2)),
                         "prob_change_location": 0.5}
          }
        """
        treasures_names = list(self.initial["treasures"].keys())
        treasures_dict = {}
        for i in range(len(treasures_names)):
            treasure = treasures_names[i]
            treasures_dict[treasure] = {
                "location": treasures_location[i],
                "possible_locations": self.initial["treasures"][treasure]["possible_locations"],
                "prob_change_location": self.initial["treasures"][treasure]["prob_change_location"]
            }
        return treasures_dict

    def generate_all_possible_states(self):
        """
        A function that generates all possible states
        :return: dictionary of all possible states for each turn to go
        for example:
        if we have 2 turns to go then the dictionary will look like this:
        {1: [state1, state2, state3, state4, state5, state6, state7, state8, state9, state10, state11, state12, state13,
             state14, state15, state16],
         2: [state1, state2, state3, state4, state5, state6, state7, state8, state9, state10, state11, state12, state13,
             state14, state15, state16],
        }
        when each state is a dictionary of the state
        !!!! it is very expensive to generate all possible states maybe we should generate them on the fly????
        """
        # initializing the all_possible_states dictionary
        all_possible_states = dict()
        # creating a key for each turn to go
        turns = self.all_turns_to_go()
        for turn in turns:
            all_possible_states[turn] = []

        # setting the constants
        optimal = self.initial["optimal"]
        infinite = self.initial["infinite"]
        game_map = self.initial["map"]

        # getting all the entities names
        pirate_ships_names = list(self.initial["pirate_ships"].keys())
        treasures_names = list(self.initial["treasures"].keys())
        marine_ships_names = list(self.initial["marine_ships"].keys())

        # getting all possible pirate locations --> list of all possible locations (positions that are not "I")
        all_possible_pirates_locations = self.all_possible_pirates_location()

        # getting all possible treasures locations --> dict {treasure_name: list of all possible locations....}
        all_possible_treasures_locations = self.all_possible_treasures_location()

        # getting all possible marine locations --> dict {marine_name: list of all possible locations....}
        all_possible_marines_locations = self.all_possible_marines_location()

        # getting all possible capacities --> dict {pirate_name: list of all possible capacities....}
        all_possible_capacities = self.all_possible_capacities()

        # getting all possible turns to go --> list of all possible turns to go
        all_possible_turns = self.all_turns_to_go()

        # getting all possible states
        for turns in all_possible_turns:
            for pirate_ships_location in all_possible_pirates_locations:
                for marine_ships_location in all_possible_marines_locations:
                    for capacities in all_possible_capacities:
                        for treasures_location in all_possible_treasures_locations:
                            pirate_ships_dict = self.generate_pirate_ships_dict(pirate_ships_location, capacities)
                            marine_ships_dict = self.generate_marine_ships_dict(marine_ships_location)
                            treasures_dict = self.generate_treasures_dict(treasures_location)
                            state = {
                                "optimal": optimal,
                                "infinite": infinite,
                                "map": game_map,
                                "pirate_ships": pirate_ships_dict,
                                "treasures": treasures_dict,
                                "marine_ships": marine_ships_dict,
                                "turns to go": turns
                            }
                            # adding the state to the all_possible_states dictionary under the key of the number of
                            # turns
                            all_possible_states[turns].append(state)
        return all_possible_states

    def get_all_possible_states(self):  ### TESTING FUNCTION
        return self.all_possible_states

    # END OF FUNCTIONS FOR CREATING ALL THE POSSIBLE STATES
    # ----------------------------------------------

    # FUNCTIONS FOR VALUE ITERATION
    # ----------------------------------------------
    def actions(self, state):
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

        turns_to_go = state["turns to go"]
        if turns_to_go == 0:
            return [(("Terminate",),)]

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

    def generate_atomic_actions_result(self, state, actions_tuple):
        """
        A function that returns the result state of the ATOMIC action on the state
        :param actions_tuple:
        :param state: dictionary of the state
        :return: state: dictionary of the result state
        IMPORTANT: the function does not check if the action is valid or not
        IMPORTANT: the function doesn't move the marine ships or the treasures this will be done in another function

        """
        new_state = deepcopy(state)
        new_state["turns to go"] -= 1
        # ATOMIC ACTIONS PARTS
        # ------------------------------------------------------
        for action in actions_tuple:

            action_type = action[0]
            ship = action[1]

            # SAIL action
            if action_type == "sail":
                new_location = action[2]
                new_state["pirate_ships"][ship]["location"] = new_location
            # End of SAIL action

            # COLLECT TREASURE action
            elif action_type == "collect_treasure":
                treasure = action[2]
                new_state["pirate_ships"][ship]["capacity"] -= 1
            # End of COLLECT TREASURE action

            # DEPOSIT TREASURE action
            elif action_type == "deposit_treasure":
                new_state["pirate_ships"][ship]["capacity"] = self.initial["pirate_ships"][ship]["capacity"]
            # End of DEPOSIT TREASURE action

            # WAIT action
            elif action_type == "wait":
                pass
            # End of WAIT action

            # TERMINATE action
            elif action_type == "Terminate":
                new_state["turns to go"] = 0
            # End of TERMINATE action

            # RESET action
            elif action_type == "reset":
                new_state = self.initial
            # End of RESET action

            # End of ATOMIC ACTIONS PARTS
            # ------------------------------------------------------
        return new_state

    def generate_single_marine_dict(self, marine, index):
        """
        A function that generates a single marine dictionary
        :param marine: the name of the marine
        :param index: the index of the marine
        :return: dictionary of the marine
        """
        marine_info = self.initial["marine_ships"][marine]
        marine_dict = {
            marine: {"index": index, "path": marine_info["path"]}
        }
        return marine_dict

    def generate_marine_possibilities(self, state):
        """
        A function that returns a list of all possible marine dictionaries
        :param state:
        :return:
        """
        marines = state["marine_ships"]
        marine_possibilities = []
        for marine in marines:
            marine_info = marines[marine]
            marine_options = []
            if len(marine_info["path"]) == 1:
                marine_options.append(self.generate_single_marine_dict(marine, 0))
            elif marine_info["index"] == 0:
                marine_options.append(self.generate_single_marine_dict(marine, 0))
                marine_options.append(self.generate_single_marine_dict(marine, 1))
            elif marine_info["index"] == len(marine_info["path"]) - 1:
                marine_options.append(self.generate_single_marine_dict(marine, len(marine_info["path"]) - 1))
                marine_options.append(self.generate_single_marine_dict(marine, len(marine_info["path"]) - 2))
            else:
                marine_options.append(self.generate_single_marine_dict(marine, marine_info["index"]))
                marine_options.append(self.generate_single_marine_dict(marine, marine_info["index"] - 1))
                marine_options.append(self.generate_single_marine_dict(marine, marine_info["index"] + 1))
            marine_possibilities.append(marine_options)
        # creating all possible combinations of the marine ships
        marine_possibilities = list(itertools.product(*marine_possibilities))
        # creating the marine ships dictionaries
        list_of_marine_dicts = []
        for possibility in marine_possibilities:
            marine_dict = {}
            for marine in possibility:
                marine_dict.update(marine)
            list_of_marine_dicts.append(marine_dict)
        return list_of_marine_dicts

    def generate_treasure_possibilities(self, state):
        """
        A function that returns a list of all possible treasure dictionaries
        :param state:
        :return:
        """
        treasures = state["treasures"]
        treasures_possibilities = []
        for treasure in treasures:
            treasure_info = treasures[treasure]
            treasure_options = []
            for location in treasure_info["possible_locations"]:
                treasure_options.append(
                    {
                        treasure: {
                            "location": location,
                            "possible_locations": treasure_info["possible_locations"],
                            "prob_change_location": treasure_info["prob_change_location"]}
                    }
                )
            treasures_possibilities.append(treasure_options)
        # creating all possible combinations of the treasures
        treasures_possibilities = list(itertools.product(*treasures_possibilities))
        # creating the treasures dictionaries
        list_of_treasures_dicts = []
        for possibility in treasures_possibilities:
            treasure_dict = {}
            for treasure in possibility:
                treasure_dict.update(treasure)
            list_of_treasures_dicts.append(treasure_dict)
        return list_of_treasures_dicts

    def generate_all_possible_outcomes(self, state, action):
        """
        A function that returns all possible outcomes of the state from taking the action
        :param state: dictionary of the current state
        :param action: tuple of the action
        :return: list of all possible outcomes of the state (list of dictionaries)
        """
        all_possible_outcomes = []
        new_state = self.generate_atomic_actions_result(state, action)
        for marine_possibility in self.generate_marine_possibilities(state):
            for treasure_possibility in self.generate_treasure_possibilities(state):
                new_state["marine_ships"].update(marine_possibility)
                new_state["treasures"].update(treasure_possibility)
                all_possible_outcomes.append(new_state)
        return all_possible_outcomes

    def generate_all_legal_outcomes(self, state):
        """
        A function that returns all possible legal outcomes of the state from taking any legal action
        :param state:
        :return: list of all possible legal outcomes of the state (list of dictionaries)
        """
        all_legal_outcomes = []
        all_actions = self.actions(state)
        for action in all_actions:
            all_legal_outcomes.extend(self.generate_all_possible_outcomes(state, action))
        return all_legal_outcomes

    def reward(self, state, action):
        """
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

        for act in action: 
            # NOTE: action is a list of tuples with specific action for each ship ((...), (...))

            if act[0] == "reset": # Tax for resetting the environment
                return -2
            
            if act[0] == "Terminate": # Tax for resetting the environment
                return 0
                
            action_type = act[0] 
            pirate_ship = act[1]
            if action_type == "deposit_treasure": # Reward for successfully retrieving a treasure to base    
                maximum_capacity_of_ship = self.initial["pirate_ships"][pirate_ship]["capacity"]
                current_capacity_of_ship = pirate_ships[pirate_ship]["capacity"]
                number_of_treasures_on_ship = maximum_capacity_of_ship - current_capacity_of_ship
                reward += (4 * number_of_treasures_on_ship)

            if action_type == "sail":
                pirate_ship_location = act[2]
            else:
                pirate_ship_location = pirate_ships[pirate_ship]["location"]
                        
            for marine in marine_ships.items(): # Penalty for encountering a marine
                marine_possibilities_probability = self.get_marine_possibilities_probability(marine)
                for item in marine_possibilities_probability.items():
                    marine_possible_location = item[0]
                    probability = item[1]
                    if marine_possible_location == pirate_ship_location:
                        reward -= 1 * probability

        return reward

    def get_marine_possibilities_probability(self, marine):
        """
        Helping function for the reward function.
        A function that given a spasific matine and returns all his possible loction next step and the probability of each one
        :param marine: a tuple reprisenting the marine
        :return: a dictionary of all possible loction next step and the probability of each one
        """
        marine_ship = marine[0]
        index = marine[1]["index"]
        path = marine[1]["path"]

        if len(path) == 1:
            return {path[0]: 1}
        
        # check if the index is 0 or the last index
        if index == 0:
            return {path[index]: 0.5, path[index + 1]: 0.5}
        
        if index == len(path) - 1:
            return {path[index]: 0.5, path[index - 1]: 0.5}
            
        else:
            return {path[index]: 1/3, path[index - 1]: 1/3, path[index + 1]: 1/3}

    def transition_probability(self, state, next_state):
        """
        A function that returns the transition probability of the action from the state to the next_state
        :param state: dictionary of the state
        :param action: tuple of the action
        :param next_state: dictionary of the next state
        :return: the transition probability of the action from the state to the next_state
        """
        marines = state["marine_ships"]
        marine_prob = 1 
        for marine in marines:
            marine_index = marines[marine]["index"]
            # marine can't really move
            if len(marines[marine]["path"]) == 1:
                continue
            # check if the index is 0 or the last index
            if marine_index == 0 or marine_index == len(marines[marine]["path"]) - 1:
                marine_prob *= 0.5
            else:
                marine_prob *= 1/3

        tresures = state["treasures"]
        next_treasures = next_state["treasures"]
        moving_treasure_prob = 1
        for treasure in tresures:
            # treasure can't really move
            if len(tresures[treasure]["possible_locations"]) == 1:
                continue
            # treasure can move - check the probability of moving
            treasure_location = tresures[treasure]["location"]
            next_treasure_location = next_treasures[treasure]["location"]
            if treasure_location != next_treasure_location:
                moving_treasure_prob *= (tresures[treasure]["prob_change_location"] * 1/len(tresures[treasure]["possible_locations"]))
            else:
                moving_treasure_prob *= (1 - tresures[treasure]["prob_change_location"] + (1/len(tresures[treasure]["possible_locations"])))

        return marine_prob * moving_treasure_prob

    def init_value_iteration(self):
        """
        A function that initializes the value iteration
        """
        v = {}
        for state in self.all_possible_states:
            v[state] = 0
        # we need to initialize the value of the terminal state to 0
    
    def value_iteration(self):
        """
        A function that runs the value iteration
        """





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
