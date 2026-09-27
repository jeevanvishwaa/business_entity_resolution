"""
Feature engineering module for pairwise entity resolution.
Computes string similarity, token overlap, numeric address agreement,
cross-field interactions, and candidate provenance features.
Supports both CompactRecord objects and standard dicts.
"""

from typing import Dict, List, Any
import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_w_ratio',
    'name_compact_ratio',
    'name_exact_match',
    'name_compact_match',
    'name_jaccard',
    'name_len_diff',
    'name_token_count_diff',
    'name_shared_token_count',
    'has_addr_s1',
    'has_addr_s2',
    'both_have_addr',
    'addr_ratio',
    'addr_partial_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    'addr_jaccard',
    'addr_shared_numbers',
    'addr_has_shared_number',
    'addr_postal_match',
    'strong_name_strong_addr',
    'strong_name_null_addr',
    'weak_name_strong_addr',
    'composite_score',
    'source_is_s2',
    'source_is_s3',
    'provenance_block_count',
    'block_name_exact',
    'block_name_compact',
    'block_name_prefix',
    'block_name_token',
    'block_addr_postal',
    'block_addr_compact'
]

def compute_pairwise_features(s1_name: dict, s1_addr: dict, 
                              target_id: str, target_rec: Any, 
                              provenance_blocks: List[str]) -> List[float]:
    """
    Computes a vector of 36 numerical features comparing S1 to a candidate target.
    target_rec can be either a CompactRecord or a dict.
    """
    # Extract target attributes
    if hasattr(target_rec, 'raw_name'):
        n2_raw = target_rec.raw_name
        n2_cmp = target_rec.compact_name
        t2_tokens = target_rec.tokens_name
        t2_core_tokens = target_rec.core_tokens_name
        a2_raw = target_rec.raw_addr
        at2_tokens = target_rec.tokens_addr
        num2_set = set(target_rec.numbers_addr)
        p2 = target_rec.postal_code
    elif isinstance(target_rec, dict) and 'norm_name' in target_rec:
        t_name = target_rec['norm_name']
        t_addr = target_rec['norm_addr']
        n2_raw = t_name['raw_clean']
        n2_cmp = t_name['compact']
        t2_tokens = t_name['tokens']
        t2_core_tokens = t_name['core_tokens']
        a2_raw = t_addr['raw_clean']
        at2_tokens = t_addr['tokens']
        num2_set = set(t_addr['numbers'])
        p2 = t_addr['postal_code']
    else:
        # Fallback empty
        n2_raw = ''
        n2_cmp = ''
        t2_tokens = []
        t2_core_tokens = []
        a2_raw = ''
        at2_tokens = []
        num2_set = set()
        p2 = ''

    # 1. Business Name Similarities
    n1_raw = s1_name['raw_clean']
    n1_cmp = s1_name['compact']

    name_ratio = fuzz.ratio(n1_raw, n2_raw)
    name_partial_ratio = fuzz.partial_ratio(n1_raw, n2_raw)
    name_token_sort_ratio = fuzz.token_sort_ratio(n1_raw, n2_raw)
    name_token_set_ratio = fuzz.token_set_ratio(n1_raw, n2_raw)
    name_w_ratio = fuzz.WRatio(n1_raw, n2_raw)
    name_compact_ratio = fuzz.ratio(n1_cmp, n2_cmp) if (n1_cmp and n2_cmp) else 0.0

    name_exact_match = 1.0 if (n1_raw and n1_raw == n2_raw) else 0.0
    name_compact_match = 1.0 if (n1_cmp and n1_cmp == n2_cmp) else 0.0

    # Token overlap
    t1_set = set(s1_name['core_tokens'])
    t2_set = set(t2_core_tokens)
    intersection = len(t1_set & t2_set)
    union = len(t1_set | t2_set)
    name_jaccard = (intersection / union) if union > 0 else 0.0
    name_len_diff = float(abs(len(n1_raw) - len(n2_raw)))
    name_token_count_diff = float(abs(len(s1_name['tokens']) - len(t2_tokens)))
    name_shared_token_count = float(intersection)

    # 2. Business Address Similarities
    a1_raw = s1_addr['raw_clean']
    has_addr_s1 = 1.0 if a1_raw else 0.0
    has_addr_s2 = 1.0 if a2_raw else 0.0
    both_have_addr = 1.0 if (has_addr_s1 and has_addr_s2) else 0.0

    if both_have_addr:
        addr_ratio = fuzz.ratio(a1_raw, a2_raw)
        addr_partial_ratio = fuzz.partial_ratio(a1_raw, a2_raw)
        addr_token_sort_ratio = fuzz.token_sort_ratio(a1_raw, a2_raw)
        addr_token_set_ratio = fuzz.token_set_ratio(a1_raw, a2_raw)

        at1_set = set(s1_addr['tokens'])
        at2_set = set(at2_tokens)
        addr_inter = len(at1_set & at2_set)
        addr_union = len(at1_set | at2_set)
        addr_jaccard = (addr_inter / addr_union) if addr_union > 0 else 0.0

        # Numeric and postal matches
        num1_set = set(s1_addr['numbers'])
        shared_nums = len(num1_set & num2_set)
        addr_shared_numbers = float(shared_nums)
        addr_has_shared_number = 1.0 if shared_nums > 0 else 0.0

        p1 = s1_addr['postal_code']
        if p1 and p2:
            addr_postal_match = 1.0 if p1 == p2 else -1.0
        else:
            addr_postal_match = 0.0
    else:
        addr_ratio = 0.0
        addr_partial_ratio = 0.0
        addr_token_sort_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_jaccard = 0.0
        addr_shared_numbers = 0.0
        addr_has_shared_number = 0.0
        addr_postal_match = 0.0

    # 3. Cross-Field Interactions
    strong_name_strong_addr = 1.0 if (name_token_set_ratio >= 85 and addr_token_set_ratio >= 85) else 0.0
    strong_name_null_addr = 1.0 if (name_token_set_ratio >= 85 and not has_addr_s2) else 0.0
    weak_name_strong_addr = 1.0 if (name_token_set_ratio < 50 and addr_token_set_ratio >= 85) else 0.0

    if both_have_addr:
        composite_score = 0.55 * name_token_set_ratio + 0.45 * addr_token_set_ratio
    else:
        composite_score = name_token_set_ratio

    # 4. Source & Provenance
    source_is_s2 = 1.0 if target_id.startswith('S2-') else 0.0
    source_is_s3 = 1.0 if target_id.startswith('S3-') else 0.0

    prov_set = set(provenance_blocks)
    provenance_block_count = float(len(prov_set))
    block_name_exact = 1.0 if 'block_name_exact' in prov_set else 0.0
    block_name_compact = 1.0 if 'block_name_compact' in prov_set else 0.0
    block_name_prefix = 1.0 if 'block_name_prefix' in prov_set else 0.0
    block_name_token = 1.0 if 'block_name_token' in prov_set else 0.0
    block_addr_postal = 1.0 if ('block_addr_postal' in prov_set or 'block_addr_street' in prov_set) else 0.0
    block_addr_compact = 1.0 if 'block_addr_compact' in prov_set else 0.0

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_w_ratio,
        name_compact_ratio,
        name_exact_match,
        name_compact_match,
        name_jaccard,
        name_len_diff,
        name_token_count_diff,
        name_shared_token_count,
        has_addr_s1,
        has_addr_s2,
        both_have_addr,
        addr_ratio,
        addr_partial_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_jaccard,
        addr_shared_numbers,
        addr_has_shared_number,
        addr_postal_match,
        strong_name_strong_addr,
        strong_name_null_addr,
        weak_name_strong_addr,
        composite_score,
        source_is_s2,
        source_is_s3,
        provenance_block_count,
        block_name_exact,
        block_name_compact,
        block_name_prefix,
        block_name_token,
        block_addr_postal,
        block_addr_compact
    ]
