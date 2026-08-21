"""Section 8: Play Style Profiles.

Pure analysis layer -- zero new fields or mechanics. Taps the existing
per-turn `current_priority` value (Section 1) into a log/histogram.
Directly comparable across trained policy, heuristic baseline, and future
archetype bots, since they all produce the same Priority sequence.
"""

from collections import Counter

from .state import Priority


class PlayStyleLog:
    def __init__(self):
        self.priorities = []
        self.deck_strengths = []

    def record(self, priority, deck_strength=None):
        self.priorities.append(priority)
        self.deck_strengths.append(deck_strength)

    def priority_distribution(self):
        """% of turns per category -- Big-Money-style skews Money/VPs,
        engine-builder skews Actions/Draw early."""
        if not self.priorities:
            return {p.value: 0.0 for p in Priority}
        counts = Counter(self.priorities)
        n = len(self.priorities)
        return {p.value: counts.get(p, 0) / n for p in Priority}

    def vps_transition_turn(self):
        """0-based turn index of the first VPs-priority turn, or None if it
        never fired (game ended before endgame threshold)."""
        for i, p in enumerate(self.priorities):
            if p == Priority.VPs:
                return i
        return None

    def vps_transition_fraction(self):
        """Same, as a fraction of the game's logged length."""
        turn = self.vps_transition_turn()
        if turn is None or not self.priorities:
            return None
        return turn / len(self.priorities)

    def transition_smoothness(self):
        """Fraction of consecutive turns where priority changed. Low =
        clean phase progression; high = flip-flopping -- a training-health
        signal, not just a style label."""
        if len(self.priorities) < 2:
            return 0.0
        changes = sum(1 for a, b in zip(self.priorities, self.priorities[1:]) if a != b)
        return changes / (len(self.priorities) - 1)

    def summary(self):
        tracked = [d for d in self.deck_strengths if d is not None]
        return {
            "priority_distribution": self.priority_distribution(),
            "vps_transition_turn": self.vps_transition_turn(),
            "vps_transition_fraction": self.vps_transition_fraction(),
            "transition_smoothness": self.transition_smoothness(),
            "turns_logged": len(self.priorities),
            "deck_strength_trajectory": self.deck_strengths,
            "final_deck_strength": tracked[-1] if tracked else None,
        }
