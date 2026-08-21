# dominion-model

A playable engine for the *Dominion* base card game, modeled in UML/SysML
(Papyrus Software Designer) and code-generated to Python, with a hand-written
reinforcement-learning layer trained against it via self-play (PPO).

The project has two halves that are deliberately kept separate:

- **The modeled game engine** (`Game`, `Player`, cards, zones, `BaseExpansion`)
  — defined in `dominion-model.uml` and generated into `generated/dominion/`.
  This is the actual rules engine: setup, turn structure, all 26 Kingdom
  cards' effects, scoring.
- **Everything that plays the game** (`demo.py`, `play_human.py`, `train.py`,
  and all of `generated/dominion/rl/`) — hand-written Python, intentionally
  *not* part of the UML model.

## Quick start

```bash
pip install -r requirements.txt

cd generated
python demo.py          # watch two heuristic AIs play a full game, headless
python play_human.py    # play interactively against a heuristic AI
python train.py --iterations 200 --rollout-steps 2048   # train a PPO agent
```

Requires Python 3.11+ (developed against 3.13/3.14) and the packages in
`requirements.txt` (`numpy`, `gymnasium`, `torch>=2.9`, `questionary`).

## Repository layout

```
dominion-model.uml           the source of truth: classes, attributes,
                              operations, and Python method bodies for the
                              modeled part of the engine
dominion-model.di / .notation / .aird   Papyrus diagram/layout metadata
                              (no game logic; safe to ignore for code changes)
generated/
  Dominion_RL_v1.0_Plan.md    the actual approved RL architecture doc that
                              generated/dominion/rl/ implements section-by-section
  demo.py                     headless 2-AI game loop, printed play-by-play
  play_human.py               interactive human-vs-AI game (questionary CLI)
  train.py                    PPO training entry point
  test_env_rollout.py         manual smoke test for the RL env's threaded handoff
  test_new_cards.py           manual smoke test for all 16 non-"First Game" Kingdom cards
  dominion/
    structure/                Game, Player, Supply, SupplyPile — modeled
    cards/                    Card, CardTypeKind, TreasureFacet, VictoryFacet — modeled
    zones/                    CardZone and its subclasses (Deck, Hand, ...) — modeled
    expansions/                BaseExpansion: the actual card data (basic +
                              Kingdom cards, recommended kingdoms) — modeled
    behavior/                 UML activity/state-machine diagrams' generated
                              stubs (ActionPhase, BuyPhase, CardLifecycle, ...)
    rl/                       reinforcement-learning layer — hand-written,
                              not modeled (see below)
```

Anything under `generated/dominion/{structure,cards,zones,expansions,behavior}/`
is derived from `dominion-model.uml` via Papyrus's pygen generator. Everything
else under `generated/` (`rl/`, `demo.py`, `play_human.py`, `train.py`, the
`test_*.py` scripts) is hand-written and has no UML counterpart.

## Editing the model

**If a change touches anything already modeled — `Game`, `Player`,
`BaseExpansion`, or any class under `structure/`, `cards/`, `zones/`,
`expansions/` — edit `dominion-model.uml` first**, then regenerate (or, if
Papyrus isn't available, hand-sync the corresponding `.py` file to match
exactly what regeneration would produce). Editing the generated `.py` first
and back-porting into the model produces drift and multi-cycle
reconcile-by-diff pain — this happened enough in this project's early history
that it's now a hard rule, not a style preference.

Changes to `demo.py`, `play_human.py`, `train.py`, or anything in
`dominion/rl/` go directly into the `.py` file — none of it has a UML
counterpart, by design (the RL layer and CLI entry points are explicitly kept
out of the model).

A few non-obvious things about how the generator actually behaves (learned
empirically, not documented by the tool itself):

- Imports are inferred only from a class's own formally-typed
  attributes/parameters, or from an explicit `<elementImport>` — never from
  plain text inside a method body.
- `__init__` has no modelable body at all; it's synthesized from the class's
  attribute list, non-composite attributes first, then composite
  single-class-typed attributes, each group in its own XML declaration order
  (not one single top-to-bottom pass). It can't carry comments.
- The generator supports nested `def` helpers with default parameters
  written as plain text inside a method body, but there's no evidence it
  supports default values on formally modeled operation parameters.
- Body text uses tab indentation for the method's own top-level statements,
  then 4 spaces per level of nesting below that.

## The card model

`BaseExpansion` (`generated/dominion/expansions/baseexpansion.py`) is the
single source of truth for card data — no card's name/cost/type is defined
anywhere else:

- `basic_cards()` — the 7 cards present in every game: Copper, Silver, Gold,
  Estate, Duchy, Province, Curse.
- `all_kingdom_cards()` — all 26 implemented Kingdom cards.
- `kingdom_card_names()` — derived from `all_kingdom_cards()`.
- `recommended_kingdoms()` — the 6 official 2nd-edition recommended kingdoms:
  First Game, Size Distortion, Deck Top, Sleight of Hand, Improvements,
  Silver & Gold.

`Game.setup(num_players)` draws a random 10-card kingdom from
`all_kingdom_cards()` by default; setting `game.kingdom_card_names` to a list
of 10 names before calling `setup()` forces a specific kingdom instead (this
is how `play_human.py`'s kingdom picker — random / suggested preset / custom
— and the recommended-kingdom presets work).

## The RL pipeline (`generated/dominion/rl/`)

Built out against `generated/Dominion_RL_v1.0_Plan.md`'s section numbering
(each module's docstring names the section it implements):

- **`state.py`** (Section 1) — `DeckState`/`PlayerState`, including the
  `current_priority` signal (Growth / Money / Shedding / VPs) that both the
  heuristic AI and the RL policy read.
- **`features.py`** (Section 2) — fixed-length per-card feature vector, built
  from card properties rather than identity, so a trained policy generalizes
  across random kingdoms instead of memorizing specific cards.
- **`reward.py`** (Section 4) — shaped reward (deck-strength delta + score-gap
  delta, weighted by turns remaining) plus a terminal win/loss signal.
- **`agents.py`** (Section 5) — baseline opponents: the heuristic baseline is
  just `Player`'s own `ai_choose_*` methods; a uniform-random baseline serves
  as a sanity floor.
- **`env.py`** (Section 6) — `DominionEnv`, a Gym-style wrapper around the
  synchronous game engine. Since `Player`'s `choose_*_fn` hooks are called
  synchronously (including deeply nested calls, e.g. Mine's trash-then-gain),
  the real game runs on a background thread and hero's (`game.players[0]`)
  hooks block on a queue handoff to turn it into a `step()`-able env —
  including off-turn cases like an opponent's Militia landing on hero
  mid-turn, which falls out of the design for free.
- **`policy.py`**, **`ppo.py`**, **`rollout.py`** — the actual policy network,
  PPO update step, and rollout/GAE collection. Custom PPO rather than
  stable-baselines3, because `DominionEnv` switches between 4 structurally
  different decision shapes per episode (card selection, pile selection,
  action choice, buy choice), which doesn't fit a single fixed Gym action
  space.
- **`evaluate.py`** — deterministic periodic evaluation against a fixed
  opponent, called from `train.py` and runnable standalone.
- **`play_style.py`** (Section 8) — pure post-hoc analysis of the
  `current_priority` sequence a game produced, directly comparable across the
  trained policy, the heuristic baseline, and future bots.

Run training with `python train.py --iterations 200 --rollout-steps 2048
--eval-every 20 --opponent heuristic` (see `train.py --help` for the full set
of PPO hyperparameters — learning rate, clip epsilon, GAE lambda, etc.).

