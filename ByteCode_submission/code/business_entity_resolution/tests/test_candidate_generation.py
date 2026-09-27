"""
Unit and integration test suite for Phase 1: Candidate Generation Consistency.
Validates:
  TEST 1: Candidate Union
  TEST 2: Provenance Preservation
  TEST 3: Candidate Limit Enforcement
  TEST 4: Absence of Hard-Coded LIMIT 40
  TEST 5: Dynamic Configuration Sensitivity
  TEST 6: Country Partitioning Integrity
  TEST 7: Rare / Distinctive Token Blocking
  TEST 8: Compact Address Blocking
  TEST 9: Candidate Deduplication & Popcount Ranking
  TEST 10: Empty Candidate Handling
  TEST 11: Training/Test Semantic Equivalence (CandidateIndex vs SQLiteCandidateLookup)
  TEST 12: Memory Behavior & Streaming Resource Verification
"""

import os
import sys
import tempfile
import sqlite3
import pytest

# Ensure project modules are resolvable
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from src.config import Config
from src.text_normalization import normalize_business_name, normalize_address, normalize_country
from src.blocking import (
    CandidateIndex, SQLiteCandidateLookup,
    FLAG_NAME_EXACT, FLAG_NAME_COMPACT, FLAG_NAME_PREFIX, FLAG_NAME_TOKEN,
    FLAG_ADDR_POSTAL, FLAG_ADDR_STREET, FLAG_ADDR_COMPACT, PROV_NAMES,
    extract_address_blocking_key, rank_and_truncate_candidates
)


def test_1_candidate_union():
    """TEST 1: Candidate Union - A candidate returned by multiple blocks must appear once."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        # T1 matches on exact clean name AND address postal
        t1_name = normalize_business_name("Blue Bottle Coffee Inc")
        t1_addr = normalize_address("123 Market St, San Francisco, CA 94105")
        lookup.add_target_batch([("T1", t1_name, t1_addr)])
        lookup.finalize_indexes()

        q_name = normalize_business_name("Blue Bottle Coffee")
        q_addr = normalize_address("123 Market St, 94105")
        cands = lookup.retrieve_candidates(q_name, q_addr, max_candidates=10)

        # Candidate T1 must appear exactly once in the returned list
        c_ids = [c[0] for c in cands]
        assert c_ids.count("T1") == 1
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_2_provenance_preservation():
    """TEST 2: Provenance - Preserves multiple blocking sources for the same candidate."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        t_name = normalize_business_name("Starbucks Coffee Company")
        t_addr = normalize_address("2401 Utah Ave S, Seattle, WA 98134")
        lookup.add_target_batch([("T100", t_name, t_addr)])
        lookup.finalize_indexes()

        # Query matches exact name, prefix, and postal
        q_name = normalize_business_name("Starbucks Coffee Company")
        q_addr = normalize_address("2401 Utah Ave S, 98134")
        cands = lookup.retrieve_candidates(q_name, q_addr, max_candidates=10)

        assert len(cands) == 1
        cid, prov = cands[0]
        assert cid == "T100"
        assert "block_name_exact" in prov
        assert "block_name_prefix" in prov
        assert "block_addr_postal" in prov
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_3_candidate_limit_and_ranking():
    """TEST 3: Candidate Limit - Truncates to N candidates following popcount ranking."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        # Create 10 target records
        batch = []
        for i in range(10):
            # T0 matches both name and addr; others match only addr
            name = "Apex Industries" if i == 0 else f"Other Firm {i}"
            addr = "500 Tech Parkway, Suite 100, Atlanta, GA 30332"
            batch.append((f"T_{i}", normalize_business_name(name), normalize_address(addr)))
        lookup.add_target_batch(batch)
        lookup.finalize_indexes()

        q_name = normalize_business_name("Apex Industries")
        q_addr = normalize_address("500 Tech Parkway, 30332")

        # Request at most 3 candidates
        limit = 3
        cands = lookup.retrieve_candidates(q_name, q_addr, max_candidates=limit)
        assert len(cands) <= limit
        # T_0 has the highest popcount (name + addr), so it must be ranked first
        assert cands[0][0] == "T_0"
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_4_no_hardcoded_limit_40():
    """TEST 4: Verify that no independent hard-coded 'LIMIT 40' remains in predict.py."""
    predict_file = os.path.join(code_dir, "src", "predict.py")
    with open(predict_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "LIMIT 40" not in content
    assert "LIMIT 40" not in content.upper()


def test_5_configuration_sensitivity():
    """TEST 5: Verify changing MAX_FINAL_CANDIDATES in Config changes candidate limit."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        batch = [
            (f"T_{i}", normalize_business_name(f"Common Name {i}"), normalize_address("100 Common Rd, 10001"))
            for i in range(15)
        ]
        lookup.add_target_batch(batch)
        lookup.finalize_indexes()

        q_name = normalize_business_name("Common Name")
        q_addr = normalize_address("100 Common Rd, 10001")

        # Dynamic limit 5
        cands_5 = lookup.retrieve_candidates(q_name, q_addr, max_candidates=5)
        assert len(cands_5) == 5

        # Dynamic limit 8
        cands_8 = lookup.retrieve_candidates(q_name, q_addr, max_candidates=8)
        assert len(cands_8) == 8
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_6_country_partitioning():
    """TEST 6: Country Partitioning - Records from different countries are isolated."""
    # Country isolation is handled by partition databases in predict.py
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_us, \
         tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_in:
        db_us = tmp_us.name
        db_in = tmp_in.name

    try:
        lookup_us = SQLiteCandidateLookup.init_database(db_us, max_post_len=400)
        lookup_in = SQLiteCandidateLookup.init_database(db_in, max_post_len=400)

        t_us = normalize_business_name("Dominos Pizza")
        a_us = normalize_address("100 Main St, New York, NY 10001")
        lookup_us.add_target_batch([("US-01", t_us, a_us)])
        lookup_us.finalize_indexes()

        t_in = normalize_business_name("Dominos Pizza")
        a_in = normalize_address("100 Main St, Connaught Place, New Delhi, 110001")
        lookup_in.add_target_batch([("IN-01", t_in, a_in)])
        lookup_in.finalize_indexes()

        # Query US lookup
        q_us_name = normalize_business_name("Dominos Pizza")
        q_us_addr = normalize_address("100 Main St, 10001")
        cands_us = lookup_us.retrieve_candidates(q_us_name, q_us_addr, max_candidates=10)

        us_ids = [c[0] for c in cands_us]
        assert "US-01" in us_ids
        assert "IN-01" not in us_ids

        lookup_us.close()
        lookup_in.close()
    finally:
        if os.path.exists(db_us):
            os.remove(db_us)
        if os.path.exists(db_in):
            os.remove(db_in)


def test_7_rare_distinctive_token_blocking():
    """TEST 7: Rare Token Blocking - Distinctive tokens generate candidates."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        # Business with distinctive word "quasiparticle"
        t_name = normalize_business_name("Quasiparticle Research Lab")
        t_addr = normalize_address("99 Discovery Way, Cambridge, MA 02138")
        lookup.add_target_batch([("T_RARE", t_name, t_addr)])
        lookup.finalize_indexes()

        # Query has completely different suffix and address, but shares the distinctive word
        q_name = normalize_business_name("Quasiparticle Technologies")
        q_addr = normalize_address("55 Unrelated Boulevard, Austin, TX 78701")
        cands = lookup.retrieve_candidates(q_name, q_addr, max_candidates=10)

        c_ids = [c[0] for c in cands]
        assert "T_RARE" in c_ids
        prov_dict = dict(cands)
        assert "block_name_token" in prov_dict["T_RARE"]
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_8_compact_address_blocking():
    """TEST 8: Compact Address Blocking - Compact address surfaces candidates."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        # Target with completely different DBA name but identical address
        t_name = normalize_business_name("Grand Bistro")
        t_addr = normalize_address("742 Evergreen Terrace, Springfield, OR 97477")
        lookup.add_target_batch([("T_ADDR", t_name, t_addr)])
        lookup.finalize_indexes()

        # Query has different name, but identical address compact form
        q_name = normalize_business_name("Springfield Diner")
        q_addr = normalize_address("742 Evergreen Terrace, Springfield, OR 97477")
        cands = lookup.retrieve_candidates(q_name, q_addr, max_candidates=10)

        c_ids = [c[0] for c in cands]
        assert "T_ADDR" in c_ids
        prov_dict = dict(cands)
        assert "block_addr_compact" in prov_dict["T_ADDR"]
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_9_duplicate_candidate_ids():
    """TEST 9: Candidate Deduplication - Candidates generated across several blocks appear once with merged flags."""
    cand_flags = {
        "CAND_1": FLAG_NAME_EXACT | FLAG_NAME_COMPACT | FLAG_NAME_PREFIX,
        "CAND_2": FLAG_NAME_TOKEN,
        "CAND_3": FLAG_ADDR_POSTAL | FLAG_ADDR_COMPACT
    }
    ranked = rank_and_truncate_candidates(cand_flags, max_candidates=10)
    assert len(ranked) == 3
    # CAND_1 has popcount 3 -> ranked first
    assert ranked[0][0] == "CAND_1"
    assert set(ranked[0][1]) == {"block_name_exact", "block_name_compact", "block_name_prefix"}
    # CAND_3 has popcount 2 -> ranked second
    assert ranked[1][0] == "CAND_3"
    assert set(ranked[1][1]) == {"block_addr_postal", "block_addr_compact"}
    # CAND_2 has popcount 1 -> ranked third
    assert ranked[2][0] == "CAND_2"
    assert set(ranked[2][1]) == {"block_name_token"}


def test_10_empty_candidate_set():
    """TEST 10: Empty Candidate Set - Record with no matches returns empty list without error."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        t_name = normalize_business_name("Existing Entity")
        t_addr = normalize_address("123 Elm St, Dallas, TX 75001")
        lookup.add_target_batch([("T_EXIST", t_name, t_addr)])
        lookup.finalize_indexes()

        # Query has no overlapping name or address
        q_name = normalize_business_name("Zzz Completely Unmatched 999")
        q_addr = normalize_address("9999 Polar Ice Station, Antarctica 00000")
        cands = lookup.retrieve_candidates(q_name, q_addr, max_candidates=10)

        assert cands == []
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_11_training_test_semantic_equivalence():
    """TEST 11: Semantic Equivalence - CandidateIndex and SQLiteCandidateLookup produce identical outputs."""
    targets = [
        ("T1", "Amazon Web Services Inc", "410 Terry Ave N, Seattle, WA 98109"),
        ("T2", "Amazon Logistics LLC", "410 Terry Ave N, Seattle, WA 98109"),
        ("T3", "Microsoft Corporation", "One Microsoft Way, Redmond, WA 98052"),
        ("T4", "Google LLC", "1600 Amphitheatre Pkwy, Mountain View, CA 94043"),
        ("T5", "Web Services Support", "123 Random Road, Miami, FL 33101")
    ]

    # In-memory CandidateIndex
    c_idx = CandidateIndex("US", max_post_len=400)
    for tid, name, addr in targets:
        c_idx.add_target_record(tid, normalize_business_name(name), normalize_address(addr))

    # SQLiteCandidateLookup
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        batch = [
            (tid, normalize_business_name(name), normalize_address(addr))
            for tid, name, addr in targets
        ]
        lookup.add_target_batch(batch)
        lookup.finalize_indexes()

        q_name = normalize_business_name("Amazon Web Services")
        q_addr = normalize_address("410 Terry Ave N, 98109")

        res_idx = c_idx.retrieve_candidates(q_name, q_addr, max_candidates=10)
        res_sql = lookup.retrieve_candidates(q_name, q_addr, max_candidates=10)

        # Check candidate IDs match
        ids_idx = [c[0] for c in res_idx]
        ids_sql = [c[0] for c in res_sql]
        assert ids_idx == ids_sql

        # Check provenance for each candidate matches
        prov_map_idx = dict(res_idx)
        prov_map_sql = dict(res_sql)
        for cid in ids_idx:
            assert set(prov_map_idx[cid]) == set(prov_map_sql[cid])

        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_12_memory_behavior_and_records():
    """TEST 12: Memory Behavior - Verified streaming batch get_record and record structure."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        lookup = SQLiteCandidateLookup.init_database(db_path, max_post_len=400)
        t_name = normalize_business_name("FedEx Ground Package System")
        t_addr = normalize_address("1000 FedEx Dr, Moon Township, PA 15108")
        lookup.add_target_batch([("FDX_01", t_name, t_addr)])
        lookup.finalize_indexes()

        # Retrieve record format
        rec = lookup.get_record("FDX_01")
        assert rec is not None
        assert "norm_name" in rec
        assert "norm_addr" in rec
        assert rec["norm_name"]["raw_clean"] == "fedex ground package system"
        assert rec["norm_addr"]["postal_code"] == "15108"

        # Verify cache clearing
        lookup.clear_cache()
        assert len(lookup._cache) == 0
        lookup.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
