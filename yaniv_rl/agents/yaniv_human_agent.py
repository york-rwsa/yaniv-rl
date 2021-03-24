import yaniv_rl.utils as utils


class YanivHumanAgent(object):
    """A human agent for yaniv. It can be used to play against trained models"""

    def __init__(self):
        """Initilize the human agent"""
        self.use_raw = True

    @staticmethod
    def step(state):
        """Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        """
        print(state["raw_obs"])
        _print_state(state["raw_obs"], state["action_record"])

        while True:
            action = input(">> You choose action (integer): ")
            try:
                action = int(action)
            except ValueError:
                print("Valid number, please")
                continue

            if action in range(len(state["legal_actions"])):
                break
            else:
                print("Illegal action, try again")

        return state["raw_legal_actions"][
            state["raw_legal_actions"].index(sorted(state["raw_legal_actions"])[action])
        ]

    def eval_step(self, state):
        """Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        """
        return self.step(state), []


def _print_state(state, action_record):
    """Print out the state of a given player

    Args:
        player (int): Player id
    """
    _action_list = []
    for i in range(1, len(action_record) + 1):
        if action_record[-i][0] == state["current_player"]:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print(">> Player", pair[0], "chose ", end="")
        _print_action(pair[1])

    print("")

    print("============= Discard pile ==============")
    print(", ".join(["  ".join([c for c in cards]) for cards in state["discard_pile"]]))

    if "pickup_top_discard" in state["legal_actions"]:
        availcards = "  ".join([c for c in state["discard_pile"][-2]])
        print(f"You can pick up: {availcards}")
    else:
        print()
    print()
    print("============== Your Hand ================")
    print("  ".join([c for c in state["hand"]]))
    print("")
    print("============= Opponents Hand ============")
    for i in range(state["player_num"]):
        if i != state["current_player"]:
            print(
                "Player {} has {} cards: {}".format(
                    i, state["hand_lengths"][i], " ".join(state["known_cards"][i])
                )
            )

    print("======== Actions You Can Choose =========")
    actions = [f"{i}:{a}" for i, a in enumerate(sorted(state["legal_actions"]))]
    print("  ".join(actions))
    print("\n")


def _print_action(action):
    print(action)