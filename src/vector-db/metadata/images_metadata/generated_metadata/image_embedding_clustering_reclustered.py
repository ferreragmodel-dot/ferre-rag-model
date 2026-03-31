#!/usr/bin/env python3
"""
Cluster Ferré images with missing PDFs using image embeddings.

Input format:
A single JSON array like:
[
  {
    "path": "...",
    "embedding": [...],
    "pdf_status": "missing"
  },
  ...
]

This script:
1. reads a JSON file of image embeddings
2. filters to pdf_status == "missing"
3. extracts season from the path
4. clusters images within each season using DBSCAN on cosine distance
5. if a cluster is too large, re-cluster it locally with stricter eps
6. writes:
   - cluster registry JSONL
   - image-to-cluster map JSON

Notes
-----
- Only missing-PDF images are clustered here.
- Each season is clustered separately.
- Noise points from DBSCAN are turned into singleton clusters.
- Cluster IDs are season-based only, e.g. FW1988-1989_cluster_031
- For missing-PDF clusters, outfit_id is set equal to cluster_id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


DEFAULT_INPUT = "embeddings_missing_from_grounded.json"
DEFAULT_CLUSTER_OUTPUT = "missing_pdf_clusters_registry.jsonl"
DEFAULT_MAP_OUTPUT = "missing_pdf_image_to_cluster_map.json"


def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON array in {path}, got {type(data).__name__}")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Expected item {i} in {path} to be an object, got {type(row).__name__}")

    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def build_cluster_id(season: str, cluster_index: int) -> str:
    return f"{season}_cluster_{cluster_index:03d}"


def extract_season_from_path(path: str) -> str:
    """
    Extract season from path.

    Expected examples:
    - dataset_datashack_2026/alta_moda_1986-87_fw/womenswear/fashion_show_photos/15600.jpg
    - dataset_datashack_2026/alta_moda_1987_ss/womenswear/fashion_show_photos/12345.jpg

    Returns normalized season strings like:
    - FW1986-1987
    - SS1987
    """
    norm_path = path.replace("\\", "/").lower()

    m_fw = re.search(r"alta_moda_(\d{4})-(\d{2})_fw", norm_path)
    if m_fw:
        start_year = m_fw.group(1)
        end_suffix = m_fw.group(2)
        end_year = start_year[:2] + end_suffix
        return f"FW{start_year}-{end_year}"

    m_ss = re.search(r"alta_moda_(\d{4})_ss", norm_path)
    if m_ss:
        year = m_ss.group(1)
        return f"SS{year}"

    raise ValueError(f"Could not extract season from path: {path}")


def run_dbscan(records: List[Dict[str, Any]], eps: float, min_samples: int) -> Dict[int, List[Dict[str, Any]]]:
    """
    Run DBSCAN on a list of records and return grouped records by label.
    """
    if not records:
        return {}

    emb_matrix = np.array([r["embedding"] for r in records], dtype=float)
    emb_matrix = normalize_embeddings(emb_matrix)

    dist_matrix = cosine_distances(emb_matrix)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[int(label)].append(records[idx])

    return grouped


def refine_cluster(
    records: List[Dict[str, Any]],
    eps: float,
    min_samples: int,
    max_cluster_size: int,
    recluster_eps_factor: float,
    max_recluster_depth: int,
    current_depth: int = 0,
) -> List[Tuple[List[Dict[str, Any]], bool]]:
    """
    Recursively re-cluster problematic large clusters using stricter eps.

    Returns:
      List of tuples: (cluster_records, reclustered_flag)
    """
    if not records:
        return []

    if len(records) <= max_cluster_size:
        return [(records, current_depth > 0)]

    if current_depth >= max_recluster_depth:
        return [([r], True) for r in records]

    stricter_eps = eps * recluster_eps_factor
    grouped = run_dbscan(records, eps=stricter_eps, min_samples=min_samples)

    final_clusters: List[Tuple[List[Dict[str, Any]], bool]] = []

    for label, group_records in grouped.items():
        if label == -1:
            continue

        if len(group_records) > max_cluster_size:
            final_clusters.extend(
                refine_cluster(
                    records=group_records,
                    eps=stricter_eps,
                    min_samples=min_samples,
                    max_cluster_size=max_cluster_size,
                    recluster_eps_factor=recluster_eps_factor,
                    max_recluster_depth=max_recluster_depth,
                    current_depth=current_depth + 1,
                )
            )
        else:
            final_clusters.append((group_records, True))

    noise_records = grouped.get(-1, [])
    for r in noise_records:
        final_clusters.append(([r], True))

    if not final_clusters:
        return [([r], True) for r in records]

    return final_clusters


def compute_cluster_metrics(cluster_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cluster_size = len(cluster_records)

    if cluster_size == 1:
        return {
            "cluster_size": 1,
            "avg_similarity": 1.0000,
            "max_pairwise_distance": 0.0000,
        }

    emb_matrix = np.array([r["embedding"] for r in cluster_records], dtype=float)
    emb_matrix = normalize_embeddings(emb_matrix)

    dist_matrix = cosine_distances(emb_matrix)
    sim_matrix = 1.0 - dist_matrix

    triu_idx = np.triu_indices(cluster_size, k=1)
    pairwise_sims = sim_matrix[triu_idx]
    pairwise_dists = dist_matrix[triu_idx]

    avg_similarity = float(np.mean(pairwise_sims)) if len(pairwise_sims) > 0 else 1.0
    max_pairwise_distance = float(np.max(pairwise_dists)) if len(pairwise_dists) > 0 else 0.0

    return {
        "cluster_size": cluster_size,
        "avg_similarity": round(avg_similarity, 4),
        "max_pairwise_distance": round(max_pairwise_distance, 4),
    }


def assign_cluster_confidence(
    cluster_size: int,
    reclustered: bool,
    avg_similarity: float,
    max_pairwise_distance: float,
) -> str:
    """
    Rule-based confidence:
    - high: singleton cluster
    - high: cluster_size <= 4 and not reclustered
    - medium: cluster_size <= 4 and reclustered
    - low: metric override for suspicious internal inconsistency
    """
    if cluster_size == 1:
        confidence = "high"
    elif cluster_size <= 4 and not reclustered:
        confidence = "high"
    elif cluster_size <= 4 and reclustered:
        confidence = "medium"
    else:
        confidence = "low"

    # Conservative metric override
    if cluster_size >= 2:
        if max_pairwise_distance > 0.35 or avg_similarity < 0.70:
            confidence = "low"

    return confidence


def cluster_one_season(
    season: str,
    season_records: List[Dict[str, Any]],
    eps: float,
    min_samples: int,
    max_cluster_size: int,
    recluster_eps_factor: float,
    max_recluster_depth: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    if not season_records:
        return [], {}

    grouped = run_dbscan(season_records, eps=eps, min_samples=min_samples)

    cluster_rows: List[Dict[str, Any]] = []
    image_to_cluster: Dict[str, str] = {}
    cluster_counter = 1

    valid_labels = sorted([lab for lab in grouped.keys() if lab != -1])
    for lab in valid_labels:
        records = grouped[lab]

        if len(records) > max_cluster_size:
            refined_clusters = refine_cluster(
                records=records,
                eps=eps,
                min_samples=min_samples,
                max_cluster_size=max_cluster_size,
                recluster_eps_factor=recluster_eps_factor,
                max_recluster_depth=max_recluster_depth,
            )
        else:
            refined_clusters = [(records, False)]

        for cluster_records, reclustered in refined_clusters:
            cluster_id = build_cluster_id(season, cluster_counter)

            metrics = compute_cluster_metrics(cluster_records)
            confidence = assign_cluster_confidence(
                cluster_size=metrics["cluster_size"],
                reclustered=reclustered,
                avg_similarity=metrics["avg_similarity"],
                max_pairwise_distance=metrics["max_pairwise_distance"],
            )

            row = {
                "cluster_id": cluster_id,
                "cluster_type": "embedding",
                "season": season,
                "outfit_id": cluster_id,
                "pdf_status": "missing",
                "pdf_paths": [],
                "image_paths": [r["path"] for r in cluster_records],
                "llm_input_mode": "image_only",
                "cluster_size": metrics["cluster_size"],
                "reclustered": reclustered,
                "avg_similarity": metrics["avg_similarity"],
                "max_pairwise_distance": metrics["max_pairwise_distance"],
                "cluster_confidence": confidence,
            }
            cluster_rows.append(row)

            for r in cluster_records:
                image_to_cluster[r["path"]] = cluster_id

            cluster_counter += 1

    noise_records = grouped.get(-1, [])
    for record in noise_records:
        cluster_id = build_cluster_id(season, cluster_counter)

        row = {
            "cluster_id": cluster_id,
            "cluster_type": "embedding",
            "season": season,
            "outfit_id": cluster_id,
            "pdf_status": "missing",
            "pdf_paths": [],
            "image_paths": [record["path"]],
            "llm_input_mode": "image_only",
            "cluster_size": 1,
            "reclustered": False,
            "avg_similarity": 1.0000,
            "max_pairwise_distance": 0.0000,
            "cluster_confidence": "high",
        }
        cluster_rows.append(row)
        image_to_cluster[record["path"]] = cluster_id
        cluster_counter += 1

    return cluster_rows, image_to_cluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster missing-PDF Ferré images using embeddings."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input JSON array with image embeddings.",
    )
    parser.add_argument(
        "--cluster-output",
        default=DEFAULT_CLUSTER_OUTPUT,
        help="Output JSONL for cluster registry.",
    )
    parser.add_argument(
        "--map-output",
        default=DEFAULT_MAP_OUTPUT,
        help="Output JSON for image-to-cluster map.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.12,
        help="First-pass DBSCAN eps on cosine distance.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples.",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Optional: cluster only one extracted season, e.g. FW1986-1987 or SS1987.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=4,
        help="If a cluster is larger than this, re-cluster it locally.",
    )
    parser.add_argument(
        "--recluster-eps-factor",
        type=float,
        default=0.7,
        help="Factor to shrink eps during local re-clustering.",
    )
    parser.add_argument(
        "--max-recluster-depth",
        type=int,
        default=1,
        help="Maximum recursion depth for re-clustering problematic clusters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("INPUT ARG:", args.input)
    print("ABS INPUT PATH:", os.path.abspath(args.input))

    rows = load_json_array(args.input)
    rows = [r for r in rows if r.get("pdf_status") == "missing"]

    if not rows:
        print("No matching rows found.")
        write_jsonl(args.cluster_output, [])
        write_json(args.map_output, {})
        return

    enriched_rows: List[Dict[str, Any]] = []
    for r in rows:
        if "path" not in r or "embedding" not in r:
            raise ValueError(f"Missing path or embedding in row: {r}")

        season = extract_season_from_path(r["path"])
        r_enriched = dict(r)
        r_enriched["season"] = season
        enriched_rows.append(r_enriched)

    if args.season:
        enriched_rows = [r for r in enriched_rows if r["season"] == args.season]

    if not enriched_rows:
        print("No matching rows found after season filtering.")
        write_jsonl(args.cluster_output, [])
        write_json(args.map_output, {})
        return

    by_season: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in enriched_rows:
        by_season[r["season"]].append(r)

    all_cluster_rows: List[Dict[str, Any]] = []
    all_image_to_cluster: Dict[str, str] = {}

    for season in sorted(by_season.keys()):
        cluster_rows, image_map = cluster_one_season(
            season=season,
            season_records=by_season[season],
            eps=args.eps,
            min_samples=args.min_samples,
            max_cluster_size=args.max_cluster_size,
            recluster_eps_factor=args.recluster_eps_factor,
            max_recluster_depth=args.max_recluster_depth,
        )
        all_cluster_rows.extend(cluster_rows)
        all_image_to_cluster.update(image_map)
        print(f"[ok] {season}: {len(by_season[season])} images -> {len(cluster_rows)} clusters")

    write_jsonl(args.cluster_output, all_cluster_rows)
    write_json(args.map_output, all_image_to_cluster)

    print(f"Wrote cluster registry: {args.cluster_output}")
    print(f"Wrote image-to-cluster map: {args.map_output}")


if __name__ == "__main__":
    main()