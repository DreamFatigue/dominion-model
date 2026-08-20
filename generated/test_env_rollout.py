"""Manual smoke test for dominion.rl.env.DominionEnv -- not a pytest suite,
just a scripted rollout to catch deadlocks/crashes in the threaded handoff."""

import random

from dominion.rl.env import DominionEnv


def random_action(mask, kind):
    legal = [i for i, m in enumerate(mask) if m]
    if kind == "cards":
        # Random legal subset (doesn't try to respect exact-count "Must"
        # semantics -- player.py degrades gracefully either way).
        k = random.randint(0, len(legal))
        chosen = set(random.sample(legal, k))
        return [1 if i in chosen else 0 for i in range(len(mask))]
    return random.choice(legal)


def main():
    env = DominionEnv(num_players=2, seed=42)
    obs, info = env.reset(seed=42)
    steps = 0
    total_reward = 0.0
    max_steps = 20000
    while steps < max_steps:
        action = random_action(info["mask"], info["kind"])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"steps={steps} terminated={terminated} total_reward={total_reward:.3f}")
    print(f"final score_gap={info.get('score_gap')} rounds={env.round_num}")
    print("play style summary:", env.play_style_log.summary())
    print("NO DEADLOCK, NO CRASH")


if __name__ == "__main__":
    main()
