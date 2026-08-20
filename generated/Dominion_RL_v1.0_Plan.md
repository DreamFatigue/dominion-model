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

**Kingdom-dependent exception:** `deck_strength`'s `w_gain` weighting (see
formula below) and the Gardens computed-value check (Section 2) both need
to know which Victory cards are in the active kingdom — i.e. `game.supply`
— not just the player's own zones. Both should call the same single
kingdom-lookup helper (e.g. "is there a count-scoring Victory pile in
supply") rather than duplicating that detection logic.

| Field | Computation |
|---|---|
| `size` | total owned card count |
| `avg_card_cost` | mean `cost` across all owned cards |
| `actions_per_turn` | **surplus only**, baseline 0 (game already grants flat +1/turn). Output of the chain simulation below. |
| `card_draw_per_turn` | **surplus only**, baseline 0 (game already grants flat 5-card draw). Output of the same chain simulation. |
| `money_generated_per_turn` | coins from treasures + coin-actions actually played within the same chain simulation. |
| `playable_cards_ratio` | `(Action + Treasure card count) / total cards` |
| `upgrade_ability` | `(Σ trash_from_hand where grants_replacement=1) / size × (1 − playable_cards_ratio)` — capacity × room-to-improve. Static, no simulation. Generalized by feature (was hardcoded to Mine/Remodel by name — now picks up any future replace-trasher, e.g. Artisan, automatically.) |
| `thinning_ability` | `(Σ trash_from_hand where grants_replacement=0) / size × (1 − playable_cards_ratio)` — pure deck-shrinking capacity (Chapel, Moneylender, Sentry's trash option), same room-to-improve weighting, same static/no-simulation treatment as `upgrade_ability`. Note: Moneylender's `trash_from_hand` is small and self-limiting (Copper-only, capped by remaining Coppers) — its main value is money, already captured in `money_generated_per_turn`, so it naturally contributes little here rather than needing a manual carve-out. |
| `attack_pressure_generated` | **surplus only**, from the same chain simulation — sum of `attack_magnitude` (Section 2) over Attack cards actually played in that pass, parallel to how `money_generated_per_turn` sums coins from the same pass. |
| `defense_density` | `Reaction card count / size` — static, no simulation, same pattern as `upgrade_ability`. |
| `gain_potential` | **surplus only**, from the same chain simulation — `buys_granted_per_turn` (surplus Buys beyond the baseline 1) + `cards_gained_per_turn` (sum of `cards_gained` over gain-cards played in the same pass). Represents expected deck-size growth this turn from any source, not just normal buys. |

**Chain simulation** (produces `actions_per_turn`, `card_draw_per_turn`,
`money_generated_per_turn`, `buys_granted_per_turn`, `cards_gained_per_turn`
together, not independently):
1. Sample/expect a 5-card hand from deck composition.
2. Always play all playable actions — never stop early, never hold back.
3. At each step, play the currently-playable Action card with the highest
   marginal value toward whichever resource (actions/draw/money) is most
   needed to keep the chain alive (support cards like Village generally
   before terminal cards like Smithy — playing a terminal first can strand
   later action cards once actions hit 0). This is a greedy value-ordered
   resolution, not full expectation-over-all-orderings.
4. Newly drawn cards mid-chain become eligible in the same pass.
5. Track running actions-remaining, cards-drawn, coins-accumulated,
   buys-accumulated, and cards-gained (Workshop/Artisan/Bureaucrat/Bandit
   -style direct gains resolved mid-chain, same as any other Action
   effect).
6. Outputs = surplus values beyond the 0/5/1 baseline (actions/draw/buys).

`avg_card_cost`'s role in any downstream formula is **deferred** — track it,
don't weight it yet.

### PlayerState
Computed per player at each decision point.

| Field | Computation |
|---|---|
| `estimated_turns_remaining` | `min` over: Province-pile depletion estimate, 3-pile depletion estimate. Each = `pile.count / avg purchases of that pile per round across all players`. Matches `Game.is_over()`'s real end conditions. **Turn-1 edge case:** with zero rounds played, the denominator is undefined (0/0) — seed with a heuristic prior (e.g. assume 1 purchase/round/player) until real purchase data accumulates, rather than dividing by zero on the very first decision point. |
| `deck_strength` | see formula below |
| `score_gap` (Current Ranking) | `self.calculate_score() − best opponent's calculate_score()`. Negative = behind, positive = ahead, 0 = tied with closest rival. (Known limitation: doesn't distinguish mid-pack positional battles — accepted for v1.) |
| `current_priority` | enum: `Actions \| Draw \| Money \| Shedding \| VPs \| Growth \| Attack \| Defense` — decision tree below |

**Deck Strength formula:**
```
deck_strength = w * (actions_per_turn + draw_per_turn + money_per_turn)
              + w_attack * attack_pressure_generated
              + w_defense * defense_density
              + w_upgrade(turns_remaining) * upgrade_ability
              + w_thin(turns_remaining) * thinning_ability
              + w_gain(turns_remaining, kingdom) * gain_potential
              # playable_cards_ratio: minor/secondary term, exact role TBD
              # avg_card_cost: deferred, no weight yet
```
- `w` — equal weight across the three chain-simulation outputs.
- `w_attack` / `w_defense` — new tunable constants, same category as `w`.
  Deliberately self-referential like the rest of `deck_strength`: they
  score your deck's offense/defense *potential*, not resolved outcomes
  against real opponents (that remains `score_gap`'s job, Section 4). No
  opponent simulation, no extra pass — `attack_pressure_generated`
  piggybacks on the chain simulation already running;
  `defense_density` is a free ratio like `upgrade_ability`.
- `w_upgrade(turns_remaining)` — scales **up** early game, decays toward 0
  late game (no time left to capitalize on trashing). Start with:
  `w_upgrade = w_slow_max * min(1, turns_remaining / T_reference)`.
- `w_thin(turns_remaining)` — same curve shape and rationale as
  `w_upgrade` (a leaner deck only pays off if there's time left to draw
  through it more often); separate tunable constant since pure thinning
  and replace-curation may not deserve equal weight.
- `w_gain(turns_remaining, kingdom)` — **not** a flat constant like the
  others. Unlike attack/defense/upgrade/thinning, raw deck growth isn't
  unconditionally good — outside a count-scoring kingdom it just dilutes
  `playable_cards_ratio`, and a flat weight risks the policy learning
  "gaining cards is always good" and hoarding junk with extra Buys even
  when there's no payoff for it (a reward-hacking failure mode, see
  Section 5). Scale `w_gain` by whether a count-scoring Victory card
  (Gardens, for now) is actually present in the episode's kingdom — near 0
  otherwise, meaningful only when it can convert into VP.
- Compute `estimated_turns_remaining` **before** `deck_strength` (dependency
  order).

**Current Priority decision tree** (check in this order — VPs override
first, since a great engine with no turns left is irrelevant):
1. **VPs** — if `estimated_turns_remaining` is low → endgame, prioritize VP
   buys regardless of anything else.
2. **Growth** — kingdom-gated (see the kingdom-dependent exception above):
   only evaluated when a count-scoring Victory card (Gardens) is in the
   active kingdom. If so, and `gain_potential` capacity is available →
   prioritize buying/gaining anything over quality, since Gardens converts
   raw card count directly to VP. Skipped entirely in kingdoms without a
   count-scoring Victory card.
3. **Defense** — if the kingdom contains Attack cards and this player's
   `defense_density` is low relative to visible/likely attack exposure →
   prioritize a Reaction card (Moat) before it's needed; reacting after
   the first hit is too late.
4. **Actions** — if `actions_per_turn` (surplus) ≈ 0 and playable action
   cards are regularly stranded in hand → buy support/action cards.
5. **Draw** — if `card_draw_per_turn` (surplus) is low relative to `size`
   → deck is diluting faster than it draws through itself.
6. **Money** — if `money_per_turn` can't reliably reach next VP tier cost
   (Duchy=5, Province=8) → buy treasure/coin generation.
7. **Attack** — if `attack_pressure_generated` ≈ 0, an affordable Attack
   card is available, and the engine (Actions/Draw/Money) is otherwise
   healthy → prioritize an Attack card for opponent-facing tempo.
8. **Shedding** — if `playable_cards_ratio` is low/dropping and
   (`upgrade_ability` > 0 or `thinning_ability` > 0) → prioritize trashing.

Thresholds are tunable; this tree is a v1 heuristic, can later be
replaced/complemented by the learned policy itself. The three new
categories (`Growth`/`Attack`/`Defense`) exist so the heuristic baseline
(Section 5) and Play Style Profiles (Section 8) can represent
Gardens-engine and attack/defense-oriented strategies — the RL policy
itself already learns these from `deck_strength`'s new terms regardless of
whether this tree captures them, since that's dense reward shaping, not
policy logic.

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
| `victory_points` | `VictoryFacet.victoryPoints` (0 if N/A) — see computed-value note below | ÷6 |
| `actions_granted` / `draw_granted` / `buys_granted` | net effect when played, 0 for non-Actions | ÷ small constant |
| `cards_gained` | net cards gained directly when played, outside the buy phase (Workshop, Artisan's gain-to-hand, Bureaucrat's Silver, Bandit's Gold), 0 for non-gainers | ÷ small constant, same treatment as `actions_granted` |
| `attack_magnitude` | hand-tuned per-card constant estimating opponent-facing disruption (Curse-gain, forced-discard, trash/topdeck, etc.), 0 for non-Attack cards | ÷ small constant, same treatment as `actions_granted` |
| `trash_from_hand` | net cards trashed **from this card's own owner's cards** when played (hand-tuned constant for variable effects, e.g. Chapel's up-to-4), 0 for non-trashers. Despite the name, applies regardless of source zone — hand (Chapel, Moneylender) or revealed (Sentry) — per Section 3's `candidate_source`; it never includes opponent-facing trash effects (e.g. Bandit trashing an opponent's Treasure), which are `attack_magnitude`'s job instead. | ÷ small constant, same treatment as `actions_granted` |
| `grants_replacement` | binary: does the trash effect also gain a card back (Mine/Remodel-style curation) vs. pure removal with nothing gained (Chapel/Moneylender-style thinning)? 0 for non-trashers | binary |
| `has_choice_effect` | binary: does playing this trigger any of the 4 decision hooks? | binary |

For **piles** (Supply), also attach a `count` field (copies remaining) —
dynamic per turn, feeds pile-race awareness (`estimated_turns_remaining`,
`VPs` priority).

**Consistency note:** DeckState's `avg_card_cost`, `playable_cards_ratio`,
`upgrade_ability`, `thinning_ability`, `attack_pressure_generated`,
`defense_density`, and `gain_potential` are all aggregates over this same
per-card feature vector — one source of truth for "what a card does,"
feeding both deck-level stats and the action space.

Normalization constants (÷8, ÷3, ÷6) are base-game maximums; revisit only
if higher-value cards get added later.

**Computed-value cards note:** `victory_points` above assumes a static
`VictoryFacet.victoryPoints` field. Gardens (Section 7, Tier 6) breaks that
assumption — its VP is `floor(total_owned_cards / 10)`, dependent on the
owning deck's `size`, not the card in isolation. Every call site that reads
`victory_points` — this feature vector *and* `calculate_score()` in
`player.py` — needs a computed-value branch, not just `VictoryFacet`
itself, or Gardens silently scores 0 wherever that branch is missing.

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
just pointed at a different zone. **Clarification:** this is not a new
parameter on `choose_cards_fn`'s signature — the caller (`play_action_card`)
already builds the `candidates` list from whichever zone before invoking
the hook (see Mine/Remodel pulling from `hand` today). `candidate_source`
just names which zone the caller pre-selected from, tracked via `context`
naming/documentation, not a runtime value threaded through the hook call
itself.

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
- **Performance note:** `Δdeck_strength` needs a full PlayerState
  computation (including Section 1's chain simulation) before *and* after
  every `step()`, and Section 3 makes `step()` fire on every atomic
  card-play/buy, not once per turn — so the chain simulation potentially
  reruns many times per turn. Profile this before committing to per-step
  recompute; consider caching or incrementally updating DeckState between
  steps that don't change deck composition (e.g. consecutive action-phase
  plays within the same chain) instead of resimulating from scratch each
  time.

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
  its Curse-gain is fully automatic. **This mechanism already exists and
  works today** — Militia (`player.py`) already calls
  `other.choose_cards_fn(...)`, the opponent's own hook, not the
  turn-holder's. Validate the wrapper's "which seat owns the pending
  decision" logic against Militia (already implemented) before building
  Bureaucrat/Bandit's routing — lower-risk than proving the mechanism for
  the first time on brand-new cards.
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
| Gardens | 4 | `VictoryFacet` needs a computed-value mode: `victoryPoints = floor(total_owned_cards / 10)`, evaluated at both `calculate_score()` time **and** wherever Section 2's `victory_points` feature is read — not a static field, and not only `calculate_score()` (see Section 2's computed-value note) |

**Feature Vector constants for the 16 new cards** (Section 2's
`attack_magnitude` / `trash_from_hand` / `grants_replacement` /
`cards_gained` — the fields that need a hand-tuned per-card value, not
derived from anything already in the game model. All other Section 2
fields — `cost`, `is_*`, `coin_value`, `victory_points`,
`actions_granted`/`draw_granted`/`buys_granted` — follow directly from
each card's rules text above and aren't repeated here):

| Card | `attack_magnitude` | `trash_from_hand` | `grants_replacement` | `cards_gained` |
|---|---|---|---|---|
| Poacher | 0 | 0 | 0 | 0 |
| Festival | 0 | 0 | 0 | 0 |
| Laboratory | 0 | 0 | 0 | 0 |
| Council Room | 0 | 0 | 0 | 0 |
| Moneylender | 0 | 1 (Copper only) | 0 → `thinning_ability` | 0 |
| Chapel | 0 | 4 (up-to, upper-bound constant) | 0 → `thinning_ability` | 0 |
| Artisan | 0 | 0 | 0 | 1 (gain-to-hand) |
| Library | 0 | 0 | 0 | 0 |
| Harbinger | 0 | 0 | 0 | 0 (relocates an owned card, doesn't add one) |
| Vassal | 0 | 0 | 0 | 0 |
| Sentry | 0 | 2 (up-to, revealed not hand — see Section 2 note) | 0 → `thinning_ability` | 0 |
| Witch | 3 (guaranteed Curse, no Reaction check bypass) | 0 | 0 | 0 |
| Bureaucrat | 1 (weakest attack — Silver-topdeck opponent counter-play exists) | 0 | 0 | 1 (Silver) |
| Bandit | 2 (can destroy an opponent's Silver/Gold) | 0 | 0 | 1 (Gold) |
| Throne Room | 0 | 0 | 0 | 0 (doubles the target card's own tags on re-invocation) |
| Gardens | N/A — Victory card, not an Action; not part of the action space | | | |

All `attack_magnitude` values above are illustrative starting points, not
derived — same "start rough, tune during training" treatment as
`w_upgrade`/`T_reference` elsewhere in this plan.

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
