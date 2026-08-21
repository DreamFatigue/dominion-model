"""Periodic evaluation against a fixed opponent, run deterministically for
reproducible numbers. Pure computation -- returns a dict; callers (train.py,
or this module's own CLI below) own console/file logging.
"""

import json

from .env import DominionEnv
from .agents import set_heuristic_baseline, set_random_baseline


def evaluate(policy, num_episodes=20, opponent_setup=set_heuristic_baseline, deterministic=True, seed=None):
    env = DominionEnv(num_players=2, opponent_setup=opponent_setup, seed=seed)

    wins = losses = ties = 0
    score_gaps, game_lengths, deck_strengths = [], [], []
    priority_dist_sums = None

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        while True:
            action, _, _ = policy.act(obs, info["mask"], info["kind"], deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        score_gap = info["score_gap"]
        if score_gap > 0:
            wins += 1
        elif score_gap < 0:
            losses += 1
        else:
            ties += 1
        score_gaps.append(score_gap)
        game_lengths.append(env.round_num)

        summary = env.play_style_log.summary()
        if summary["final_deck_strength"] is not None:
            deck_strengths.append(summary["final_deck_strength"])
        dist = summary["priority_distribution"]
        if priority_dist_sums is None:
            priority_dist_sums = {k: 0.0 for k in dist}
        for k, v in dist.items():
            priority_dist_sums[k] += v

    n = num_episodes
    return {
        "num_episodes": n,
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "tie_rate": ties / n,
        "avg_score_gap": sum(score_gaps) / n,
        "avg_game_length": sum(game_lengths) / n,
        "avg_final_deck_strength": (sum(deck_strengths) / len(deck_strengths)) if deck_strengths else None,
        "avg_priority_distribution": {k: v / n for k, v in priority_dist_sums.items()},
    }


if __name__ == "__main__":
    import argparse
    import torch

    from .policy import DominionPolicy

    parser = argparse.ArgumentParser(description="Evaluate a trained DominionPolicy checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    policy = DominionPolicy()
    policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()

    opponent_setup = set_heuristic_baseline if args.opponent == "heuristic" else set_random_baseline
    result = evaluate(policy, num_episodes=args.num_episodes, opponent_setup=opponent_setup, seed=args.seed)
    print(json.dumps(result, indent=2))
