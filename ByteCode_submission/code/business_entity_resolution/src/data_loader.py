"""
Data loading and country-partitioning utilities.
Efficiently loads, partitions, and caches TSV records.
"""

import os
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from src.text_normalization import normalize_business_name, normalize_address, normalize_country

def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Loads train_ground_truth.tsv into {s1_id: set(matched_ids)}."""
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                gt_map[s1_id] = {mid.strip() for mid in parts[1].split(',') if mid.strip()}
            else:
                gt_map[s1_id] = set()
    return gt_map

def load_source1(s1_path: str) -> List[dict]:
    """Loads Source 1 records with normalized fields."""
    records = []
    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            entity_id = parts[0].strip()
            b_name = parts[1].strip()
            b_addr = parts[2].strip()
            country = normalize_country(parts[3].strip())

            records.append({
                'entity_id': entity_id,
                'business_name': b_name,
                'business_address': b_addr,
                'country': country,
                'norm_name': normalize_business_name(b_name),
                'norm_addr': normalize_address(b_addr)
            })
    return records
