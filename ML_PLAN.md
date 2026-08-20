# Wiring in the ML/RL agent

Current state: `player.py` has hardcoded heuristics, not a learned policy. The
seams below are where a real policy plugs in once the state representation
and deck analyzer are settled.

## Decision points (all in `generated/dominion/structure/player.py`)

- `Player.choose_cards_fn(candidates, prompt, context, count)` — set per-player,
  called for Cellar discards, Mine/Remodel trash targets, Militia discards.
- `Player.choose_pile_fn(candidates, prompt, context)` — called for
  Workshop/Mine/Remodel gains.
- Buy-phase pile selection — currently inline in `generated/demo.py:play_turn`
  (`random.choice(tier)`), not a hook. Needs to be pulled out into its own
  `choose_buy_fn` on `Player` before an RL agent can override it, the same
  way `choose_cards_fn`/`choose_pile_fn` already work.
- Action-play order — also inline in `demo.py:play_turn` (`action_cards[0]`).
  Same deal: needs its own hook (`choose_action_fn`) before it's swappable.

## Once state + deck analyzer are defined

1. Turn the deck analyzer's output into the observation vector fed to the
   policy (whatever fields he settles on — counts by card, coins/actions/buys
   remaining, supply pile counts, etc.).
2. Implement the 4 hooks above as thin wrappers: encode state → call
   policy → decode action back into a candidate/pile choice.
3. Reward signal: `Player.calculate_score()` already computes VP; use
   score delta (or win/loss at game end) as the reward.
4. Self-play loop: reuse `demo.py`'s game loop minus the `print`/`pause`
   calls — swap both players' hooks to the policy and run headless for
   training, keep the printing version for human-readable playtesting.

No ML library is imported anywhere yet — pick one when he's ready
(PyTorch is the natural default for anything RL-shaped).
