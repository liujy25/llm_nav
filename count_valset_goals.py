#!/usr/bin/env python3
"""
Count goal categories in a Habitat val split dataset.

Usage examples:
  python count_valset_goals.py --config benchmark/nav/pointnav/pointnav_hm3d.yaml
  python count_valset_goals.py --dataset-path data/datasets/objectnav/hm3d/v1/val/val.json.gz
  python count_valset_goals.py --config benchmark/nav/pointnav/pointnav_hm3d.yaml --split val
"""

import argparse
import gzip
import json
import os
import glob
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _load_json_maybe_gz(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_goal_name_from_episode(ep: Dict[str, Any]) -> Optional[str]:
    # Most ObjectNav datasets store goals as list[dict] with object_category.
    goals = ep.get("goals")
    if isinstance(goals, list) and len(goals) > 0 and isinstance(goals[0], dict):
        g0 = goals[0]
        for key in ("object_category", "category", "name"):
            v = g0.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # Fallback: episode-level fields.
    for key in ("object_category", "goal_category", "goal"):
        v = ep.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Last fallback: category id.
    for key in ("object_category_id", "goal_category_id"):
        v = ep.get(key)
        if isinstance(v, int):
            return f"id:{v}"

    return None


def _collect_dataset_files(dataset_path: str) -> List[str]:
    path = os.path.expanduser(dataset_path)

    # Glob pattern support.
    if any(ch in path for ch in "*?[]"):
        files = sorted(glob.glob(path, recursive=True))
        return [p for p in files if os.path.isfile(p)]

    # File path.
    if os.path.isfile(path):
        return [path]

    # Directory path: scan JSON/JSON.GZ files recursively.
    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.json.gz"), recursive=True)
        )
        return [p for p in files if os.path.isfile(p)]

    return []


def _expand_content_scene_files(index_data: Dict[str, Any], index_file: str) -> List[str]:
    """
    Expand Habitat content scene files from an index json/json.gz.
    Typical fields:
      - content_scenes_path: "{data_path}/content/{scene}.json.gz"
      - content_scenes: ["scene1", "scene2", ...] or ["*"]
    """
    base_dir = os.path.dirname(os.path.abspath(index_file))
    content_scenes = index_data.get("content_scenes", None)
    content_scenes_path = index_data.get("content_scenes_path", None)

    if content_scenes_path is None and content_scenes is None:
        return []

    # Normalize content_scenes into a list.
    scenes: List[str] = []
    if isinstance(content_scenes, list):
        scenes = [str(s).strip() for s in content_scenes if str(s).strip()]
    elif isinstance(content_scenes, str) and content_scenes.strip():
        scenes = [content_scenes.strip()]

    # Resolve template/path.
    template = content_scenes_path if isinstance(content_scenes_path, str) else "content/{scene}.json.gz"
    template = template.replace("{data_path}", base_dir)
    if not os.path.isabs(template):
        template = os.path.join(base_dir, template)

    # Wildcard mode.
    if len(scenes) == 0 or scenes == ["*"]:
        wildcard = template.replace("{scene}", "*")
        files = sorted(glob.glob(wildcard))
        return [p for p in files if os.path.isfile(p)]

    # Explicit scene names.
    out: List[str] = []
    for scene in scenes:
        candidates = [
            template.replace("{scene}", scene),
            template.replace("{scene}", os.path.splitext(os.path.basename(scene))[0]),
        ]
        found = False
        for c in candidates:
            if os.path.isfile(c):
                out.append(c)
                found = True
                break
        if found:
            continue

        # Fallback: glob by scene stem under template directory.
        stem = os.path.splitext(os.path.basename(scene))[0]
        glob_pattern = template.replace("{scene}", stem + "*")
        for p in sorted(glob.glob(glob_pattern)):
            if os.path.isfile(p):
                out.append(p)
                found = True
                break

    # De-duplicate while preserving order.
    dedup: List[str] = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


def _resolve_dataset_path_from_config(config_path: str, split_override: Optional[str]) -> Tuple[str, str]:
    """
    Resolve dataset data_path and split from Habitat config.
    Requires habitat-lab installed in environment.
    """
    try:
        import habitat  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import habitat. Please install habitat-lab or pass --dataset-path directly."
        ) from e

    cfg = habitat.get_config(config_path)
    split = split_override or str(cfg.habitat.dataset.split)
    data_path = str(cfg.habitat.dataset.data_path)

    # Handle common template styles.
    data_path = data_path.replace("{split}", split)
    data_path = data_path.replace("${split}", split)

    return data_path, split


def main() -> None:
    parser = argparse.ArgumentParser(description="Count loaded valset goal categories")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Habitat config path (e.g. benchmark/nav/pointnav/pointnav_hm3d.yaml)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset JSON/JSON.GZ path, glob pattern, or directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split override when resolving from config (default: use config value)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Only print top-k categories (0 means all)",
    )
    args = parser.parse_args()

    if args.config is None and args.dataset_path is None:
        raise SystemExit("Please provide either --config or --dataset-path")

    if args.dataset_path is not None:
        dataset_path = args.dataset_path
        split_name = args.split or "unknown"
    else:
        dataset_path, split_name = _resolve_dataset_path_from_config(args.config, args.split)

    files = _collect_dataset_files(dataset_path)
    if not files:
        raise SystemExit(
            f"No dataset files found. Resolved path: {dataset_path}\n"
            "Tip: pass --dataset-path explicitly or check --config/--split."
        )

    total_episodes = 0
    missing_goal = 0
    counter: Counter = Counter()
    used_files = []
    visited_files = set()
    pending_files = list(files)

    while pending_files:
        fpath = pending_files.pop(0)
        if fpath in visited_files:
            continue
        visited_files.add(fpath)
        try:
            data = _load_json_maybe_gz(fpath)
        except Exception:
            continue

        episodes = data.get("episodes", [])
        if not isinstance(episodes, list):
            episodes = []

        # Handle split index files where episodes are in content/* scene files.
        if len(episodes) == 0:
            content_files = _expand_content_scene_files(data, fpath)
            for cf in content_files:
                if cf not in visited_files:
                    pending_files.append(cf)
            continue

        used_files.append(fpath)
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            total_episodes += 1
            goal_name = _extract_goal_name_from_episode(ep)
            if goal_name is None:
                missing_goal += 1
                continue
            counter[goal_name] += 1

    if total_episodes == 0:
        raise SystemExit(
            "No episodes parsed from dataset files. "
            "Please verify dataset path/split and file format."
        )

    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    if args.topk > 0:
        items = items[: args.topk]

    print("=" * 72)
    print("Valset Goal Category Statistics")
    print("=" * 72)
    print(f"split                 : {split_name}")
    print(f"resolved_dataset_path : {dataset_path}")
    print(f"files_scanned         : {len(visited_files)}")
    print(f"files_used            : {len(used_files)}")
    print(f"episodes_total        : {total_episodes}")
    print(f"episodes_missing_goal : {missing_goal}")
    print(f"unique_goal_categories: {len(counter)}")
    print("-" * 72)

    for name, cnt in items:
        ratio = 100.0 * cnt / max(1, total_episodes)
        print(f"{name:<24} {cnt:>8}  ({ratio:6.2f}%)")

    if len(used_files) > 0:
        print("-" * 72)
        print("Sample used files:")
        for p in used_files[:10]:
            print(f"  {p}")
        if len(used_files) > 10:
            print(f"  ... ({len(used_files) - 10} more)")


if __name__ == "__main__":
    main()
