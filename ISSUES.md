# Issues

- [x] Sort cards by type and price when buying — buy-phase piles are now sorted (Action, then Treasure, then Victory, then Curse; cost ascending within each) in both `demo.py` and `play_human.py`.
- [ ] GUI? — open question, not implemented; still CLI-only (`questionary`-driven).
- [x] The Vassal subroutine needs the same rules for print display — Vassal now prints what it reveals, and what it plays/discards, the same as a normal action play.
- [x] Show victory points after any score card is bought — buying a Victory/Curse pile now prints the buyer's current score.
- [x] Defense should never be priority unless there's a defense card — `Priority.Defense` now also requires the kingdom to actually contain a Reaction card (`KingdomContext.has_defense_card`), not just an Attack card.
- [x] Do curses count toward ending score — yes, already correct: `Player.calculate_score()` includes `CardTypeKind.Curse` cards (each is -1 VP). No change needed.
- [x] Player fails on Throne Room — fixed. The human player never had `choose_action_fn` set (only the AI did), so choosing a Throne Room target crashed with `AttributeError: 'Player' object has no attribute 'choose_action_fn'`. `play_human.py` now defines `human_choose_action` and wires it up.
- [x] Attacks should give control to the other players to respond to them — Moat blocking used to be a silent, hardcoded auto-block (`declineMoatReveal`, which nothing ever set). It's now a real decision routed through the same `choose_cards_fn` hook every other card decision uses, for both the human (prompted) and AI (always reveals, matching prior behavior).
- [x] I skipped a Bureaucrat's attack — same root cause as above: when the target held a Moat, the attack (and any prompt from it, like Bureaucrat's topdeck choice) was silently skipped with zero feedback. Fixed by the same change.
- [x] Test mode with every card possible in hand and 2nd copies in draw, unlimited actions and buys, to test all cards — added `generated/test_all_cards.py`.

## Player fails on Throne Room (original traceback, kept for reference)

```
=== YOUR TURN (Player 0) ===
    Hand: ['Market', 'Moat', 'Estate', 'Smithy', 'Throne Room']
    Actions: 1  Buys: 1  Coins: 0
? Play which action card? Throne Room (cost 4)
Traceback (most recent call last):
  File "C:\Users\n80th\VS Code\dominion-model\model\play_human.py", line 149, in <module>
    main()
    ~~~~^^
  File "C:\Users\n80th\VS Code\dominion-model\model\play_human.py", line 129, in main
    human_turn(human, 0, game)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\n80th\VS Code\dominion-model\model\play_human.py", line 60, in human_turn
    p.play_action_card(card, game)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "C:\Users\n80th\VS Code\dominion-model\model\dominion\structure\player.py", line 136, in play_action_card
    self._resolve_action_effect(card, game)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "C:\Users\n80th\VS Code\dominion-model\model\dominion\structure\player.py", line 420, in _resolve_action_effect
    target = self.choose_action_fn(action_cards_in_hand,
             ^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Player' object has no attribute 'choose_action_fn'. Did you mean: 'choose_cards_fn'?
```
