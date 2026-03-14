#!/usr/bin/env python3
"""
Batch Evaluation System for LLM Navigation
Implements VLFM-style automated evaluation while preserving the original client logic
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import tqdm

import habitat
from habitat.config.default import get_config


class EpisodeLogger:
    """Manages episode-level logging with VLFM-style structure"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def is_evaluated(self, episode_id: int, scene_id: str) -> bool:
        """Check if episode has been evaluated"""
        filename = self._get_filename(episode_id, scene_id)
        if not filename.exists():
            return False
        # Check if file is not empty
        return filename.stat().st_size > 0

    def log_episode(self, episode_id: int, scene_id: str, data: Dict[str, Any]) -> None:
        """Log episode results to JSON file"""
        filename = self._get_filename(episode_id, scene_id)

        # Skip if already exists and not empty
        if self.is_evaluated(episode_id, scene_id):
            print(f"[Logger] Episode {episode_id} in {scene_id} already logged, skipping")
            return

        log_data = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }

        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"[Logger] Logged episode {episode_id:04d} to {filename.name}")

    def _get_filename(self, episode_id: int, scene_id: str) -> Path:
        """Generate filename for episode log"""
        scene_name = Path(scene_id).stem
        return self.log_dir / f"{episode_id:04d}_{scene_name}.json"

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Load all episode results from log directory"""
        results = []
        for json_file in sorted(self.log_dir.glob("*.json")):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data:  # Skip empty files
                        results.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Logger] Warning: Failed to load {json_file}: {e}")
        return results


class BatchEvaluator:
    """Batch evaluation system for LLM navigation"""

    def __init__(
        self,
        config_path: str,
        server_url: str,
        log_dir: str,
        split: str = "val",
        max_episodes: int = -1,
        resume: bool = True,
        **client_kwargs
    ):
        """
        Args:
            config_path: Path to habitat config
            server_url: LLM server URL
            log_dir: Directory for episode logs
            split: Dataset split (train/val/test)
            max_episodes: Maximum episodes to evaluate (-1 for all)
            resume: Skip already evaluated episodes
            **client_kwargs: Additional arguments for the client
        """
        self.config_path = config_path
        self.server_url = server_url
        self.log_dir = log_dir
        self.split = split
        self.max_episodes = max_episodes
        self.resume = resume
        self.client_kwargs = client_kwargs

        self.logger = EpisodeLogger(log_dir)

        # Load dataset to get episode list
        self.config = get_config(config_path)

        # Get episode list
        self.episodes = self._load_episodes(split)

        print(f"[BatchEval] Loaded {len(self.episodes)} episodes from {split} split")
        if max_episodes > 0:
            self.episodes = self.episodes[:max_episodes]
            print(f"[BatchEval] Limited to {len(self.episodes)} episodes")

    def _load_episodes(self, split: str) -> List[Dict[str, Any]]:
        """Load episode list from dataset"""
        # Create a config copy with the correct split
        import copy
        from omegaconf import OmegaConf

        config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        config.habitat.dataset.split = split

        # Create temporary env to access dataset
        with habitat.Env(config=config) as env:
            dataset = env._dataset
            episodes = []
            for episode in dataset.episodes:
                episodes.append({
                    'episode_id': episode.episode_id,
                    'scene_id': episode.scene_id,
                    'start_position': episode.start_position,
                    'start_rotation': episode.start_rotation,
                    'goals': episode.goals if hasattr(episode, 'goals') else []
                })
        return episodes

    def run_evaluation(self) -> Dict[str, Any]:
        """Run batch evaluation on all episodes"""
        # Import here to avoid circular dependency
        from habitat_llm_nav_client import HabitatLLMNavClient, create_habitat_env

        # Filter episodes if resuming
        episodes_to_eval = []
        for ep in self.episodes:
            scene_name = Path(ep['scene_id']).stem
            if self.resume and self.logger.is_evaluated(int(ep['episode_id']), scene_name):
                print(f"[BatchEval] Skipping episode {ep['episode_id']} (already evaluated)")
            else:
                episodes_to_eval.append(ep)

        print(f"[BatchEval] Evaluating {len(episodes_to_eval)} episodes")

        if len(episodes_to_eval) == 0:
            print("[BatchEval] No episodes to evaluate!")
            return self._compute_statistics()

        # Statistics tracking
        num_success = 0
        num_total = 0

        # Progress bar
        pbar = tqdm.tqdm(total=len(episodes_to_eval), desc="Evaluating episodes")

        # Create environment once (don't pass scene_path to avoid dataset reload issues)
        print("[BatchEval] Creating Habitat environment...")
        env = create_habitat_env(self.config_path, scene_path=None)

        for ep_info in episodes_to_eval:
            episode_id = int(ep_info['episode_id'])
            scene_id = ep_info['scene_id']
            scene_name = Path(scene_id).stem

            pbar.set_description(f"Episode {episode_id:04d} ({scene_name})")

            try:
                # Create output directory for this episode
                episode_output_dir = Path(self.log_dir) / f"episode_{episode_id:04d}"
                episode_output_dir.mkdir(exist_ok=True)

                # Create client with the shared environment
                # goal_text=None will make client auto-load from episode
                client = HabitatLLMNavClient(
                    env=env,
                    server_url=self.server_url,
                    goal_text=None,  # Auto-load from episode
                    episode_id=episode_id,
                    output_dir=str(episode_output_dir),
                    visualize=False,  # Disable visualization for batch eval
                    **self.client_kwargs
                )

                # Run navigation
                success = client.run_navigation()

                # Collect metrics
                metrics = {
                    'success': success,
                    'keyframes': client.keyframe_count,
                    'pointnav_steps': client.pointnav_step_total,
                }

                # Add habitat metrics if available
                if client.episode_metrics:
                    metrics.update({
                        'habitat_success': client.episode_metrics.get('success', False),
                        'spl': client.episode_metrics.get('spl', 0.0),
                        'soft_spl': client.episode_metrics.get('soft_spl', 0.0),
                        'distance_to_goal': client.episode_metrics.get('distance_to_goal', -1),
                        'num_steps': client.episode_metrics.get('num_steps', -1)
                    })

                # Add video path if video was saved
                video_path = episode_output_dir / "navigation_video.mp4"
                if video_path.exists():
                    metrics['video_path'] = str(video_path)

                # Add trajectory path
                trajectory_path = episode_output_dir / "trajectory.json"
                if trajectory_path.exists():
                    metrics['trajectory_path'] = str(trajectory_path)

                # Log episode
                self.logger.log_episode(episode_id, scene_name, metrics)

                # Update statistics
                num_total += 1
                if metrics.get('habitat_success', success):
                    num_success += 1

                success_rate = num_success / num_total * 100
                pbar.set_postfix({'success_rate': f'{success_rate:.1f}%'})

            except Exception as e:
                print(f"\n[BatchEval] Error evaluating episode {episode_id}: {e}")
                import traceback
                traceback.print_exc()

                # Log failure
                self.logger.log_episode(episode_id, scene_name, {
                    'success': False,
                    'error': str(e),
                    'habitat_success': False,
                    'spl': 0.0
                })

            pbar.update(1)

        pbar.close()

        # Close environment
        env.close()

        # Compute final statistics
        stats = self._compute_statistics()

        return stats

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics from all logged episodes"""
        results = self.logger.get_all_results()

        if len(results) == 0:
            print("[BatchEval] No results to compute statistics")
            return {}

        # Aggregate metrics
        metrics_list = defaultdict(list)
        for result in results:
            for key in ['success', 'habitat_success', 'spl', 'soft_spl',
                       'distance_to_goal', 'keyframes', 'pointnav_steps']:
                if key in result:
                    metrics_list[key].append(result[key])

        # Compute averages
        stats = {}
        for key, values in metrics_list.items():
            if len(values) > 0:
                if key in ['success', 'habitat_success']:
                    stats[f'{key}_rate'] = np.mean(values) * 100  # Convert to percentage
                else:
                    stats[f'avg_{key}'] = np.mean(values)
                    stats[f'std_{key}'] = np.std(values)

        stats['num_episodes'] = len(results)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch Evaluation for LLM Navigation (VLFM-style)"
    )

    # Dataset configuration
    parser.add_argument("--config", type=str,
                       default="benchmark/nav/pointnav/pointnav_hm3d.yaml",
                       help="Path to habitat config")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--max_episodes", type=int, default=-1,
                       help="Maximum episodes to evaluate (-1 for all)")

    # Server configuration
    parser.add_argument("--server", type=str, default="http://10.19.126.158:1874",
                       help="LLM navigation server URL")

    # Logging configuration
    parser.add_argument("--log_dir", type=str, default="./eval_results",
                       help="Directory for episode logs")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't skip already evaluated episodes")

    # Client configuration
    parser.add_argument("--max_pointnav_steps", type=int, default=500,
                       help="Maximum PointNav steps per episode")
    parser.add_argument("--camera_height", type=float, default=1.5,
                       help="Camera height above base")
    parser.add_argument("--use_pointnav", action="store_true",
                       help="Use PointNav policy for navigation")
    parser.add_argument("--pointnav_policy", type=str, default="data/pointnav_weights.pth",
                       help="Path to PointNav policy weights")
    parser.add_argument("--save_video", action="store_true",
                       help="Save videos for each episode")

    args = parser.parse_args()

    # Create evaluator
    evaluator = BatchEvaluator(
        config_path=args.config,
        server_url=args.server,
        log_dir=args.log_dir,
        split=args.split,
        max_episodes=args.max_episodes,
        resume=not args.no_resume,
        max_pointnav_steps=args.max_pointnav_steps,
        camera_height=args.camera_height,
        use_pointnav=args.use_pointnav,
        pointnav_policy_path=args.pointnav_policy,
        save_video=args.save_video
    )

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting Batch Evaluation")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Server: {args.server}")
    print(f"Log directory: {args.log_dir}")
    print(f"Resume: {not args.no_resume}")
    print(f"{'='*60}\n")

    start_time = time.time()
    stats = evaluator.run_evaluation()
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Episodes evaluated: {stats.get('num_episodes', 0)}")

    if stats:
        print(f"\n--- Aggregate Metrics ---")
        if 'habitat_success_rate' in stats:
            print(f"Success Rate: {stats['habitat_success_rate']:.2f}%")
        if 'avg_spl' in stats:
            print(f"SPL: {stats['avg_spl']:.4f} ± {stats.get('std_spl', 0):.4f}")
        if 'avg_soft_spl' in stats:
            print(f"Soft SPL: {stats['avg_soft_spl']:.4f} ± {stats.get('std_soft_spl', 0):.4f}")
        if 'avg_distance_to_goal' in stats:
            print(f"Distance to Goal: {stats['avg_distance_to_goal']:.2f}m ± {stats.get('std_distance_to_goal', 0):.2f}m")
        if 'avg_keyframes' in stats:
            print(f"Avg Keyframes: {stats['avg_keyframes']:.1f}")
        if 'avg_pointnav_steps' in stats:
            print(f"Avg PointNav Steps: {stats['avg_pointnav_steps']:.1f}")

    # Save summary
    summary_path = Path(args.log_dir) / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'config': args.config,
            'split': args.split,
            'server': args.server,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_time_minutes': elapsed_time / 60,
            'statistics': stats
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Individual episode logs in: {args.log_dir}")
    print(f"\nTo analyze results, run:")
    print(f"  python scripts/parse_llm_nav_results.py {args.log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
