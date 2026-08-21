"""Top-level PPO training entry point for DominionEnv -- trains hero (seat 0)
against a fixed baseline opponent (heuristic by default). See
Dominion_RL_v1.0_Plan.md Sections 5/6 and the approved training plan for the
architecture (custom PPO, not stable-baselines3 -- DominionEnv's info["kind"]
switches between 4 structurally different decision shapes per episode, which
doesn't fit a single fixed gym action space).

Usage:
    python train.py --iterations 200 --rollout-steps 2048 --eval-every 20
"""

import argparse
import json
import os
import time

import torch

from dominion.rl.env import DominionEnv
from dominion.rl.policy import DominionPolicy
from dominion.rl.rollout import collect_rollout
from dominion.rl.ppo import ppo_update
from dominion.rl.evaluate import evaluate
from dominion.rl.agents import set_heuristic_baseline, set_random_baseline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume policy weights from.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dominion", "rl", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    eval_log_path = os.path.join(run_dir, "eval_log.jsonl")

    opponent_setup = set_heuristic_baseline if args.opponent == "heuristic" else set_random_baseline

    env = DominionEnv(num_players=2, opponent_setup=opponent_setup, seed=args.seed)
    policy = DominionPolicy()
    if args.resume:
        policy.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed weights from {args.resume}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs, info = env.reset(seed=args.seed)

    print(f"Run: {run_name}  (dir: {run_dir})")
    print(f"Opponent: {args.opponent}  Iterations: {args.iterations}  Rollout steps: {args.rollout_steps}")

    for iteration in range(1, args.iterations + 1):
        buffer, obs, info = collect_rollout(env, policy, obs, info, args.rollout_steps,
                                             gamma=args.gamma, lam=args.lam)
        stats = ppo_update(policy, optimizer, buffer, clip_eps=args.clip_eps, epochs=args.epochs,
                            minibatch_size=args.minibatch_size, vf_coef=args.vf_coef, ent_coef=args.ent_coef)

        mean_reward = sum(buffer.rewards) / len(buffer.rewards)
        num_episodes_in_rollout = sum(buffer.dones)
        print(
            f"[iter {iteration:4d}] mean_reward={mean_reward:+.4f} "
            f"episodes={num_episodes_in_rollout:3d} "
            f"policy_loss={stats['policy_loss']:+.4f} value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.3f} approx_kl={stats['approx_kl']:.4f} "
            f"clip_frac={stats['clip_fraction']:.3f}"
        )

        torch.save(policy.state_dict(), os.path.join(run_dir, "policy_latest.pt"))

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            checkpoint_path = os.path.join(run_dir, f"policy_iter{iteration}.pt")
            torch.save(policy.state_dict(), checkpoint_path)

            eval_random = evaluate(policy, num_episodes=args.eval_episodes, opponent_setup=set_random_baseline,
                                    deterministic=True, seed=1000 + iteration)
            eval_heuristic = evaluate(policy, num_episodes=args.eval_episodes, opponent_setup=set_heuristic_baseline,
                                       deterministic=True, seed=2000 + iteration)
            print(f"  eval@{iteration} vs random:    win_rate={eval_random['win_rate']:.2f} "
                  f"avg_score_gap={eval_random['avg_score_gap']:+.2f} avg_len={eval_random['avg_game_length']:.1f}")
            print(f"  eval@{iteration} vs heuristic: win_rate={eval_heuristic['win_rate']:.2f} "
                  f"avg_score_gap={eval_heuristic['avg_score_gap']:+.2f} avg_len={eval_heuristic['avg_game_length']:.1f}")

            with open(eval_log_path, "a") as f:
                f.write(json.dumps({
                    "iteration": iteration,
                    "train_stats": stats,
                    "mean_reward": mean_reward,
                    "vs_random": eval_random,
                    "vs_heuristic": eval_heuristic,
                }) + "\n")

    print("Training complete.")


if __name__ == "__main__":
    main()
