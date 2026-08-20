# Dominion RL Agent — v1.0 Implementation Plan

Single consolidated plan. Supersedes all prior individual design docs.
Written for direct implementation against `dominion-model.uml` /
`generated/dominion/structure/player.py` / `demo.py`.

**Repo context:** Game logic already exists (`Card`, `Player`, `Supply`,
`Game`, `CardTypeKind`). `Player` already has `choose_cards_fn(candidates,
prompt, context, count)` and `choose_pile_fn(candidates, prompt, context)`
as pluggable hooks (see `demo.py` / the human CLI harness). `choose_buy_fn`
and `choose_action_fn` do **not** exist yet — currently hardcoded as
`random.choice(tier)` and `action_cards[0]` in `demo.py`. Building those two
real hooks is a prerequisite for everything below.

**Suggested build order for tonight:**
1. Add `choose_buy_fn` / `choose_action_fn` hooks to `Player` (pull the
   hardcoded logic out of `demo.py`, matching the existing hook pattern)
2. DeckState + PlayerState computation (Section 1)
3. Card Feature Vector (Section 2)
4. Action Space + masking (Section 3)
5. Reward Signal (Section 4)
6. Gym-style env wrapper (Section 6) — wires 1–5 together
7. Self-play / baseline opponents (Section 5)
8. New kingdom card content (Section 7) — can happen in parallel, needed
   before the "real" training run but not before the pipeline works
9. Play Style Profiles (Section 8) — free, do last, pure logging

---

## 1. DeckState & PlayerState

### DeckState
Computed over everything a player owns: `deck + hand + discardPile +
playArea`.

| Field | Computation |
|---|---|
| `size` | total owned card count |
| `avg_card_cost` | mean `cost` across all owned cards |
| `actions_per_turn` | **surplus only**, baseline 0 (game already grants flat +1/turn). Output of the chain simulation below. |
| `card_draw_per_turn` | **surplus only**, baseline 0 (game already grants flat 5-card draw). Output of the same chain simulation. |
| `money_generated_per_turn` | coins from treasures + coin-actions actually played within the same chain simulation. |
| `playable_cards_ratio` | `(Action + Treasure card count) / total cards` |
| `upgrade_ability` | `(Mine + Remodel count / size) × (1 − playable_cards_ratio)` — capacity × room-to-improve. Static, no simulation. |

**Chain simulation** (produces `actions_per_turn`, `card_draw_per_turn`,
`money_generated_per_turn` together, not independently):
1. Sample/expect a 5-card hand from deck composition.
2. Always play all playable actions — never stop early, never hold back.
3. At each step, play the currently-playable Action card with the highest
   marginal value toward whichever resource (actions/draw/money) is most
   needed to keep the chain alive (support cards like Village generally
   before terminal cards like Smithy — playing a terminal first can strand
   later action cards once actions hit 0). This is a greedy value-ordered
   resolution, not full expectation-over-all-orderings.
4. Newly drawn cards mid-chain become eligible in the same pass.
5. Track running actions-remaining, cards-drawn, coins-accumulated.
6. Outputs = surplus values beyond the 0/5 baseline.

`avg_card_cost`'s role in any downstream formula is **deferred** — track it,
don't weight it yet.

### PlayerState
Computed per player at each decision point.

| Field | Computation |
|---|---|
| `estimated_turns_remaining` | `min` over: Province-pile depletion estimate, 3-pile depletion estimate. Each = `pile.count / avg purchases of that pile per round across all players`. Matches `Game.is_over()`'s real end conditions. |
| `deck_strength` | see formula below |
| `score_gap` (Current Ranking) | `self.calculate_score() − best opponent's calculate_score()`. Negative = behind, positive = ahead, 0 = tied with closest rival. (Known limitation: doesn't distinguish mid-pack positional battles — accepted for v1.) |
| `current_priority` | enum: `Actions \| Draw \| Money \| Shedding \| VPs` — decision tree below |

**Deck Strength formula:**
```
deck_strength = w * (actions_per_turn + draw_per_turn + money_per_turn)
              + w_upgrade(turns_remaining) * upgrade_ability
              # playable_cards_ratio: minor/secondary term, exact role TBD
              # avg_card_cost: deferred, no weight yet
```
- `w` — equal weight across the three chain-simulation outputs.
- `w_upgrade(turns_remaining)` — scales **up** early game, decays toward 0
  late game (no time left to capitalize on trashing). Start with:
  `w_upgrade = w_slow_max * min(1, turns_remaining / T_reference)`.
- Compute `estimated_turns_remaining` **before** `deck_strength` (dependency
  order).

**Current Priority decision tree** (check in this order — VPs override
first, since a great engine with no turns left is irrelevant):
1. **VPs** — if `estimated_turns_remaining` is low → endgame, prioritize VP
   buys regardless of anything else.
2. **Actions** — if `actions_per_turn` (surplus) ≈ 0 and playable action
   cards are regularly stranded in hand → buy support/action cards.
3. **Draw** — if `card_draw_per_turn` (surplus) is low relative to `size`
   → deck is diluting faster than it draws through itself.
4. **Money** — if `money_per_turn` can't reliably reach next VP tier cost
   (Duchy=5, Province=8) → buy treasure/coin generation.
5. **Shedding** — if `playable_cards_ratio` is low/dropping and
   `upgrade_ability` > 0 → prioritize trashing.

Thresholds are tunable; this tree is a v1 heuristic, can later be
replaced/complemented by the learned policy itself.

---

## 2. Card Feature Vector

Every card — in hand, in a pile, revealed from deck — is represented by the
**same fixed-length feature vector** built from its properties, not its
identity. This is what lets the policy generalize across random kingdoms.

| Feature | Source | Encoding |
|---|---|---|
| `cost` | `Card.cost` | ÷8 |
| `is_action` / `is_treasure` / `is_victory` / `is_curse` / `is_attack` / `is_reaction` | `CardTypeKind in types` | binary each |
| `coin_value` | `TreasureFacet.coinValue` (0 if N/A) | ÷3 |
| `victory_points` | `VictoryFacet.victoryPoints` (0 if N/A) | ÷6 |
| `actions_granted` / `draw_granted` / `buys_granted` | net effect when played, 0 for non-Actions | ÷ small constant |
| `has_choice_effect` | binary: does playing this trigger any of the 4 decision hooks? | binary |

For **piles** (Supply), also attach a `count` field (copies remaining) —
dynamic per turn, feeds pile-race awareness (`estimated_turns_remaining`,
`VPs` priority).

**Consistency note:** DeckState's `avg_card_cost`, `playable_cards_ratio`,
and `upgrade_ability` are aggregates over this same per-card feature
vector — one source of truth for "what a card does," feeding both
deck-level stats and the action space.

Normalization constants (÷8, ÷3, ÷6) are base-game maximums; revisit only
if higher-value cards get added later.

---

## 3. Action Space

**Per-decision-point stepping** — one RL `step()` = one atomic decision
(play one card, buy one pile), not a whole-turn combined action.
Treasure-playing (`play_treasures()`) is automatic, not a decision point.

**Sizing:**
- `HAND_SLOTS` — fixed generous cap sized to theoretical max deck size
  (derived from total supply at setup). Unused slots masked out.
- `PILE_SLOTS` — fixed at max Kingdom-pile count (17 standard: 7 basic + 10
  Kingdom). Slot→pile assignment is per-episode/random; the *feature
  vector* riding on each slot carries the real information (see Section 2),
  not the slot index — this is what makes random-kingdom training work.

**May vs. Must masking rule** (applies to any multi-select decision):
- **May** (e.g. Cellar: any number) — STOP/commit always legal, including
  empty selection.
- **Must** (e.g. Mine/Remodel: exactly 1; Militia: exactly `excess`) — STOP
  masked illegal until the required count is reached.
- Driven dynamically by `context` + `count` already present in the real
  `choose_cards_fn` signature — not a hardcoded property of the space.

**The 4 decision points:**

| Hook | Space | Mask | STOP legal? | Loop |
|---|---|---|---|---|
| `choose_action_fn` (new) | categorical(`HAND_SLOTS`+1) | slot holds playable Action card, `actions>0` | always | repeat until STOP or none playable |
| `choose_buy_fn` (new) | categorical(`PILE_SLOTS`+1) | `pile.count>0`, `cost<=coins` | always | repeat until STOP or none affordable |
| `choose_cards_fn` (exists) | multi-binary(`HAND_SLOTS`) + commit | slot ∈ `candidates` | May/Must rule above | toggle-then-commit sequence |
| `choose_pile_fn` (exists) | categorical(`PILE_SLOTS`+1) | slot ∈ pre-filtered `candidates` (`eligible_piles()`) | always | single-shot |

**Generalized candidate sources** (needed for the 16 new cards, Section 7):
`choose_cards_fn`'s candidates aren't always the hand — extend with a
`candidate_source` tag: `hand`, `discard` (Harbinger), `revealed` (Vassal /
Library / Sentry, 1–2 top-of-deck cards). Same multi-binary+commit shape,
just pointed at a different zone.

**Sentry's 3-way choice** — decompose into 3 sequential
`choose_cards_fn` calls over the same 2 revealed cards, no new shape needed:
1. `context="sentry_trash"` (May, 0–2)
2. `context="sentry_discard"` (May, 0–remaining)
3. `context="sentry_reorder"` (simple keep-order/swap binary for ≤2 leftover cards)

---

## 4. Reward Signal

Built as a function of PlayerState fields — reuses the same
`estimated_turns_remaining` phase logic already driving Deck Strength's
`w_upgrade`, so shaping and priority stay consistent.

```
R(t) = R_shaped(t) + R_terminal          # R_terminal = 0 except final step

R_shaped(t) = w_engine(turns_remaining) * Δdeck_strength
            + w_score(turns_remaining)  * Δscore_gap

R_terminal = sign(final score_gap)       # or scaled by margin — test both
```
- `w_engine` high early, decays late (same curve shape as `w_upgrade`).
- `w_score` low early, rises late (mirror curve: `w_score = 1 - w_engine`).
- Start with linear curves as a function of `turns_remaining / T_reference`,
  tune during training.
- Watch for scale mismatch between dense shaped reward and sparse terminal
  reward — likely needs a global scale factor once real training runs
  start.

---

## 5. Self-Play / Opponent Setup

- **One shared policy controls all seats** in a training game (start fixed
  at **2 players** — simplest, matches `Game.setup(2)`; extend to variable
  player count later).
- **Snapshot pool** — periodically save policy snapshots; sample some
  training-game seats from the pool (uniform sampling to start) instead of
  always the live policy, to avoid strategy collapse / forgetting.
- **Baselines (for evaluation, not self-play population):**
  - **Heuristic** — existing `ai_choose_cards`/`ai_choose_pile` logic in the
    UML model, plus simple greedy versions of the new `choose_buy_fn`
    (cheapest-affordable-that's-not-Copper, or similar) and
    `choose_action_fn` (hand-order or simple priority) to match the current
    `demo.py` style.
  - **Random** — uniform legal action sampling, respecting masks. Sanity
    floor.
- **Eval loop**, separate from the training pool: every N iterations, run
  fixed batches vs. heuristic, vs. random, vs. an early self-snapshot. Track
  win rate, final `score_gap`, game length, `deck_strength` trajectory — not
  just the reward curve (reward-only monitoring can hide reward hacking).

---

## 6. Gym-Style Env Wrapper

Standard interface: `obs = env.reset()`, `obs, reward, done, info =
env.step(action)`.

**`reset()`:**
1. `Game()` + `Game.setup(num_players)` (2 for now).
2. Assign `choose_cards_fn`/`choose_pile_fn`/`choose_buy_fn`/
   `choose_action_fn` per seat (live policy / snapshot / baseline, per
   Section 5's sampling rule).
3. Auto-resolve opponent turns until the trained policy's seat hits its
   first decision point.
4. Return initial `obs` + action mask.

**`step(action)`:**
1. Decode `action` per whichever of the 4 decision types is pending
   (Section 3), respecting the mask.
2. Apply it by calling the real hook with the decoded choice.
3. If that phase/loop isn't done for this seat, next `obs` = next step of
   the same phase, same seat.
4. Else, auto-resolve subsequent opponent turns (their own assigned hooks)
   until control returns to the trained policy or the game ends.
5. Compute reward (Section 4) from that seat's PlayerState delta.
6. Check `done` via `Game.is_over()`; apply terminal reward if so.
7. Return `(obs, reward, done, info)`.

**Key rules:**
- One `step()` = one decision point, not one turn.
- Opponent turns are resolved synchronously inside `step()`/`reset()`, never
  exposed as separate steps — keeps this single-agent-shaped for standard
  RL libraries (SB3, RLlib, CleanRL) despite the underlying multi-seat game.
- Action mask must be exposed as part of `obs`/`info`, not just applied
  internally — needed for masked-action algorithms (e.g. SB3's
  `MaskablePPO`).

**Two updates required for the full 26-card kingdom (Section 7):**
- **Off-turn decision routing** — Bureaucrat/Bandit force decisions onto
  *opponents*, not the turn-holder. `step()`'s "who owns the next pending
  decision" must be based on which seat the game engine is currently
  asking, not whose turn it is. If that seat is policy-controlled, yield
  control to it mid-turn with its own PlayerState/mask; if
  baseline/snapshot-controlled, resolve internally. Reward attribution
  still follows the normal per-seat delta rule. Witch needs no routing —
  its Curse-gain is fully automatic.
- **Nested invocation (Throne Room)** — target-card selection reuses
  `choose_action_fn` unchanged; the environment then invokes that card's
  full effect (including its own decision points) **twice in sequence**, as
  a re-entrant sub-routine call. No Action Space changes — same shapes
  exercised twice.

---

## 7. Kingdom Card Pool — Full 26-Card Base Set

Currently only 10 Kingdom cards exist in `Game.setup()`, always the same
10, unconditionally added — so "random kingdom" currently has nothing to
randomize. Base Dominion (2nd Ed.) has **26 Kingdom cards total**. Add the
remaining 16, then randomize `setup()`'s selection (10-of-26 per game, ~5.3M
combinations).

**Already modeled (10):** Village, Smithy, Market, Workshop, Moat, Cellar,
Merchant, Militia, Mine, Remodel.

**To add (16), grouped by implementation tier:**

**Tier 1 — direct fit, drop into existing `elif name==...` chain, no new hooks:**
| Card | Cost | Effect |
|---|---|---|
| Poacher | 4 | +1 Card, +1 Action, +1 Coin; discard 1 per empty supply pile |
| Festival | 5 | +2 Actions, +1 Buy, +2 Coins |
| Laboratory | 5 | +2 Cards, +1 Action |
| Council Room | 5 | +4 Cards, +1 Buy; each other player draws 1 |
| Moneylender | 4 | Trash a Copper from hand for +3 Coins |

**Tier 2 — reuses `choose_cards_fn`, new `context` values, existing shape:**
| Card | Cost | Effect | New context(s) |
|---|---|---|---|
| Chapel | 2 | Trash up to 4 cards from hand | `chapel_trash` (May, ≤4) |
| Artisan | 6 | Gain a card to hand costing ≤5; put a card from hand onto deck | `artisan_gain` then `artisan_topdeck` |
| Library | 5 | Draw until 7 cards in hand; may skip drawn Actions | `library_skip` (May, per drawn Action, repeated) |

**Tier 3 — new candidate source (discard pile / revealed cards):**
| Card | Cost | Effect | New context(s) |
|---|---|---|---|
| Harbinger | 3 | +1 Card, +1 Action; may put a card from discard onto deck | `harbinger_topdeck` (May, source=discard) |
| Vassal | 3 | +2 Coins; discard top of deck, may play it if Action | `vassal_play` (May, source=revealed) |
| Sentry | 5 | +1 Card, +1 Action; look at top 2, trash/discard/reorder | `sentry_trash` → `sentry_discard` → `sentry_reorder` (source=revealed) |

**Tier 4 — Attack cards, extend Militia's Moat-check pattern:**
| Card | Cost | Effect | Decision needed? |
|---|---|---|---|
| Witch | 5 | +2 Cards; each unblocked opponent gains a Curse | None — automatic |
| Bureaucrat | 4 | Gain Silver onto own deck; each unblocked opponent topdecks a Victory card | Opponent decision if multiple Victory cards — `bureaucrat_topdeck`, off-turn routing |
| Bandit | 5 | Gain a Gold; each unblocked opponent reveals top 2, trashes a non-Copper Treasure, discards rest | Opponent decision if 2+ eligible — `bandit_trash`, off-turn routing |

**Tier 5 — structural mechanism, no new content beyond target selection:**
| Card | Cost | Requirement |
|---|---|---|
| Throne Room | 4 | `choose_action_fn` target selection, then re-entrant double invocation (Section 6 wrapper update) |

**Tier 6 — scoring-only, no action space impact:**
| Card | Cost | Requirement |
|---|---|---|
| Gardens | 4 | `VictoryFacet` needs a computed-value mode: `victoryPoints = floor(total_owned_cards / 10)` at `calculate_score()` time, not a static field |

**Suggested build order:** Tier 1 → Tier 6 (isolated) → Tier 2 → Tier 4
(proves off-turn routing, needed regardless) → Tier 3 → Tier 5 (last,
depends on every other card working as a callable sub-routine).

**Recommended preset kingdoms** (rulebook, useful as structured
training/eval kingdoms alongside fully-random draws):
- **First Game:** Cellar, Market, Merchant, Militia, Mine, Moat, Remodel, Smithy, Village, Workshop *(current default)*
- **Size Distortion:** Artisan, Bandit, Bureaucrat, Chapel, Festival, Gardens, Sentry, Throne Room, Witch, Workshop
- **Deck Top:** Artisan, Bureaucrat, Council Room, Festival, Harbinger, Laboratory, Moneylender, Sentry, Vassal, Village
- **Sleight of Hand:** Cellar, Council Room, Festival, Gardens, Library, Harbinger, Militia, Poacher, Smithy, Throne Room
- **Improvements:** Artisan, Cellar, Market, Merchant, Mine, Moat, Moneylender, Poacher, Remodel, Witch
- **Silver & Gold:** Bandit, Bureaucrat, Chapel, Harbinger, Laboratory, Merchant, Mine, Moneylender, Throne Room, Vassal

**Why this matters now vs. later:** training against the current fixed
10-card kingdom validates the pipeline (wrapper, reward, self-play) but
produces a policy overfit to one kingdom. Treat the full pool as a
prerequisite for the *final* training run, not a blocker for building and
testing everything else first.

---

## 8. Play Style Profiles (free — do last)

Pure analysis layer, zero new fields or mechanics. `Current Priority`
(Section 1) is already logged once per turn per player. A "play style
profile" is just the aggregated shape of that sequence over a game:
- **Priority distribution** — % of turns per category (Big-Money-style
  skews `Money`/`VPs`; engine-builder skews `Actions`/`Draw` early).
- **Transition timing** — when the shift to `VPs` happens (turn number or
  fraction of `estimated_turns_remaining`).
- **Transition smoothness** — frequent flip-flopping vs. clean phase
  progression; a training-health signal, not just a style label.

Log alongside the eval metrics in Section 5's evaluation loop. Directly
comparable across trained policy, heuristic baseline, and future archetype
bots since they all produce the same `Current Priority` sequence. No
implementation dependency on anything else — just tap the existing
per-turn `current_priority` value into a log/histogram.

---

## Summary Checklist

- [ ] Add `choose_buy_fn` / `choose_action_fn` hooks to `Player`
- [ ] DeckState computation (chain simulation + static ratios)
- [ ] PlayerState computation (turns remaining → deck strength → priority)
- [ ] Card feature vector (shared by DeckState aggregates + action slots)
- [ ] Action space + masks (4 hooks, May/Must rule, generalized candidate sources)
- [ ] Reward signal (shaped + terminal, PlayerState-driven)
- [ ] Gym env wrapper (`reset`/`step`, off-turn routing, nested invocation)
- [ ] Self-play pool + heuristic/random baselines + eval loop
- [ ] Add 16 new Kingdom cards (Tiers 1→6) + randomize `setup()`'s kingdom draw
- [ ] Play style profile logging
