"""Section 4: Reward Signal.

R(t) = R_shaped(t) + R_terminal
R_shaped(t) = w_engine(turns_remaining) * delta_deck_strength
            + w_score(turns_remaining)  * delta_score_gap
R_terminal  = sign(final score_gap), only on the terminal step.
"""

import numpy as np

from .state import T_REFERENCE

# Dense shaped reward and sparse terminal reward are on different natural
# scales; this is the "likely needs a global scale factor" knob the plan
# calls out -- start at 1.0, revisit once real training runs start.
SHAPED_REWARD_SCALE = 1.0


def w_engine(turns_remaining):
    """High early, decays toward 0 late -- same curve shape as w_upgrade."""
    return min(1.0, turns_remaining / T_REFERENCE)


def w_score(turns_remaining):
    """Mirror of w_engine: low early, rises to 1 late."""
    return 1.0 - w_engine(turns_remaining)


def compute_reward(prev_player_state, next_player_state, done):
    """prev/next are PlayerState for the acting seat, taken immediately
    before and after the decision that just resolved (Section 6 step 5:
    'from that seat's PlayerState delta')."""
    turns_remaining = next_player_state.estimated_turns_remaining
    delta_strength = next_player_state.deck_strength - prev_player_state.deck_strength
    delta_score_gap = next_player_state.score_gap - prev_player_state.score_gap

    shaped = SHAPED_REWARD_SCALE * (
        w_engine(turns_remaining) * delta_strength
        + w_score(turns_remaining) * delta_score_gap
    )

    terminal = 0.0
    if done:
        terminal = float(np.sign(next_player_state.score_gap))

    return shaped + terminal
