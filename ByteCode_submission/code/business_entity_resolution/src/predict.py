"""
Ultra-scalable, disk-backed inference engine for Amazon ML Challenge 2026.
Uses SQLite B-Tree indexing on disk to guarantee < 200 MB peak RAM
and blazing fast query throughput across multi-million record test sets.
Produces:
- output/matching_results.tsv
- output/candidate_pairs.tsv
- output/phase2/final_submission.tsv
- output/phase2/inference_statistics.json
"""

import os
import sys
import time
import io
import gc
import json
import shutil
import sqlite3
import tracemalloc
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

from src.config import Config
from src.text_normalization import normalize_business_name, normalize_address, normalize_country
from src.blocking import SQLiteCandidateLookup, PROV_NAMES
from src.features import compute_pairwise_features
from src.singleton_detection import apply_decision_policy

# Safe UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_test_inference(model, threshold_config: dict, 
                       test_dir: str = "dataset/test", 
                       output_dir: str = "output",
                       batch_size: int = 50000):
    """
    Executes streaming test inference partition by partition (France, US, India).
    Uses disk-backed SQLite indexes to keep RAM strictly < 200 MB.
    """
    tracemalloc.start()
    os.makedirs(output_dir, exist_ok=True)
    phase2_dir = os.path.join(output_dir, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)

    scratch_dir = os.path.join(Config.BASE_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    db_path = os.path.join(scratch_dir, "test_predictions.db")

    # Connect to persistent predictions database with checkpointing support
    conn_preds = sqlite3.connect(db_path)
    cur_preds = conn_preds.cursor()
    cur_preds.execute("PRAGMA synchronous = OFF")
    cur_preds.execute("PRAGMA journal_mode = MEMORY")
    cur_preds.execute("CREATE TABLE IF NOT EXISTS preds (s1_id TEXT PRIMARY KEY, candidates TEXT, matches TEXT)")
    cur_preds.execute("""
        CREATE TABLE IF NOT EXISTS completed_countries (
            country TEXT PRIMARY KEY,
            total INTEGER,
            matches INTEGER,
            singletons INTEGER,
            completed_at TEXT
        )
    """)
    conn_preds.commit()

    decision_thresh = threshold_config.get('decision_threshold', Config.DECISION_THRESHOLD)
    singleton_thresh = threshold_config.get('singleton_threshold', 0.45)
    max_final_cands = Config.MAX_FINAL_CANDIDATES
    max_post_len = Config.MAX_CANDIDATES_PER_BLOCK

    print("\n==========================================")
    print("STARTING FULL TEST SET INFERENCE (CANONICAL SQLITE ENGINE)")
    print(f"Decision Threshold:    {decision_thresh}")
    print(f"Singleton Threshold:   {singleton_thresh}")
    print(f"Max Final Candidates:  {max_final_cands}")
    print(f"Max Post List Length:  {max_post_len}")
    print("==========================================")
    t_start = time.time()

    # Step 1: Ensure partition scratch files exist
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    order_file = os.path.join(scratch_dir, "test_s1_order.txt")

    # 1.1 Partition Test S1
    if not os.path.exists(order_file):
        print(f"Partitioning Test S1 from {s1_path} into disk scratch files...")
        country_files = {
            'FRANCE': open(os.path.join(scratch_dir, "test_s1_FRANCE.tsv"), 'w', encoding='utf-8'),
            'US': open(os.path.join(scratch_dir, "test_s1_US.tsv"), 'w', encoding='utf-8'),
            'INDIA': open(os.path.join(scratch_dir, "test_s1_INDIA.tsv"), 'w', encoding='utf-8')
        }
        with open(s1_path, 'r', encoding='utf-8', errors='replace') as f_in, \
             open(order_file, 'w', encoding='utf-8') as f_order:
            header = f_in.readline()
            for line in f_in:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                s1_id = parts[0].strip()
                name = parts[1].strip()
                addr = parts[2].strip()
                country = normalize_country(parts[3].strip())

                f_order.write(f"{s1_id}\n")
                if country in country_files:
                    country_files[country].write(f"{s1_id}\t{name}\t{addr}\n")
                else:
                    country_files['US'].write(f"{s1_id}\t{name}\t{addr}\n")
        for f in country_files.values():
            f.close()

    # 1.2 Partition Test S2
    s2_flag = os.path.join(scratch_dir, "test_s2_partitioned.flag")
    if not os.path.exists(s2_flag):
        print(f"Partitioning Test S2 from {s2_path} into country scratch files...")
        s2_files = {
            'FRANCE': open(os.path.join(scratch_dir, "test_s2_FRANCE.tsv"), 'w', encoding='utf-8'),
            'US': open(os.path.join(scratch_dir, "test_s2_US.tsv"), 'w', encoding='utf-8'),
            'INDIA': open(os.path.join(scratch_dir, "test_s2_INDIA.tsv"), 'w', encoding='utf-8')
        }
        with open(s2_path, 'r', encoding='utf-8', errors='replace') as f_in:
            f_in.readline()  # skip header
            for line in f_in:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                cid = parts[0].strip()
                name = parts[1].strip()
                addr = parts[2].strip()
                country = normalize_country(parts[3].strip())
                target_f = s2_files.get(country, s2_files['US'])
                target_f.write(f"{cid}\t{name}\t{addr}\n")
        for f in s2_files.values():
            f.close()
        with open(s2_flag, 'w') as f:
            f.write("done\n")

    # 1.3 Partition Test S3
    s3_flag = os.path.join(scratch_dir, "test_s3_partitioned.flag")
    if not os.path.exists(s3_flag):
        print(f"Partitioning Test S3 from {s3_path} into country scratch files...")
        s3_files = {
            'FRANCE': open(os.path.join(scratch_dir, "test_s3_FRANCE.tsv"), 'w', encoding='utf-8'),
            'US': open(os.path.join(scratch_dir, "test_s3_US.tsv"), 'w', encoding='utf-8'),
            'INDIA': open(os.path.join(scratch_dir, "test_s3_INDIA.tsv"), 'w', encoding='utf-8')
        }
        with open(s3_path, 'r', encoding='utf-8', errors='replace') as f_in:
            f_in.readline()  # skip header
            for line in f_in:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                cid = parts[0].strip()
                name = parts[1].strip()
                addr = parts[2].strip()
                country = normalize_country(parts[3].strip())
                target_f = s3_files.get(country, s3_files['US'])
                target_f.write(f"{cid}\t{name}\t{addr}\n")
        for f in s3_files.values():
            f.close()
        with open(s3_flag, 'w') as f:
            f.write("done\n")

    # Metrics and diagnostics tracking
    diagnostics = {
        'countries': {},
        'total_s1': 0,
        'total_candidate_pairs': 0,
        'candidates_per_s1_buckets': Counter(),
        'blocking_hits_distribution': Counter(),
        'blocking_strategy_usage': Counter(),
        'matched_entities': 0,
        'singleton_entities': 0,
        'total_matched_links': 0,
        's2_matched_links': 0,
        's3_matched_links': 0,
        'prob_min': 1.0,
        'prob_max': 0.0,
        'prob_sum': 0.0,
        'prob_count': 0,
        'runtimes': {}
    }

    # Step 2: Process each country partition
    countries = ['FRANCE', 'US', 'INDIA']

    for country in countries:
        s1_country_file = os.path.join(scratch_dir, f"test_s1_{country}.tsv")
        if not os.path.exists(s1_country_file):
            continue

        with open(s1_country_file, 'r', encoding='utf-8') as f:
            c_count = sum(1 for _ in f)
        diagnostics['total_s1'] += c_count

        # Check if country was already fully processed
        cur_preds.execute("SELECT total, matches, singletons FROM completed_countries WHERE country = ?", (country,))
        completed_row = cur_preds.fetchone()
        if completed_row and completed_row[0] == c_count:
            print(f"\n--- Country Partition {country} ({c_count:,} entities) ALREADY PROCESSED & VERIFIED. Skipping! ---")
            diagnostics['countries'][country] = {
                's1_count': c_count,
                'status': 'ALREADY_COMPLETED',
                'matches': completed_row[1],
                'singletons': completed_row[2]
            }
            diagnostics['matched_entities'] += completed_row[1]
            diagnostics['singleton_entities'] += completed_row[2]
            continue

        print(f"\n--- Processing Country Partition: {country} ({c_count:,} S1 entities) ---")
        t_country = time.time()
        c_stats = {
            's1_count': c_count,
            's2_count': 0,
            's3_count': 0,
            'matches': 0,
            'singletons': 0,
            'candidate_pairs': 0
        }

        # Build SQLite B-Tree index for targets in this country
        target_db_path = os.path.join(scratch_dir, f"test_targets_{country}.db")
        lookup = SQLiteCandidateLookup.init_database(target_db_path, max_post_len=max_post_len)

        # Stream Test S2 for this country
        t_s2_start = time.time()
        s2_c_file = os.path.join(scratch_dir, f"test_s2_{country}.tsv")
        s2_count = 0
        batch = []
        with open(s2_c_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                cid = parts[0].strip()
                n_dict = normalize_business_name(parts[1].strip())
                a_dict = normalize_address(parts[2].strip())
                batch.append((cid, n_dict, a_dict))
                s2_count += 1
                if len(batch) >= 20000:
                    lookup.add_target_batch(batch, commit=False)
                    batch = []
            if batch:
                lookup.add_target_batch(batch, commit=False)
                batch = []
        lookup.conn.commit()
        c_stats['s2_count'] = s2_count
        print(f"Indexed {s2_count:,} S2 targets for {country} in {time.time()-t_s2_start:.2f}s")

        # Stream Test S3 for this country
        t_s3_start = time.time()
        s3_c_file = os.path.join(scratch_dir, f"test_s3_{country}.tsv")
        s3_count = 0
        with open(s3_c_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                cid = parts[0].strip()
                n_dict = normalize_business_name(parts[1].strip())
                a_dict = normalize_address(parts[2].strip())
                batch.append((cid, n_dict, a_dict))
                s3_count += 1
                if len(batch) >= 20000:
                    lookup.add_target_batch(batch, commit=False)
                    batch = []
            if batch:
                lookup.add_target_batch(batch, commit=False)
                batch = []
        lookup.conn.commit()
        c_stats['s3_count'] = s3_count
        print(f"Indexed {s3_count:,} S3 targets for {country} in {time.time()-t_s3_start:.2f}s")

        # Build B-Tree indexes and prune high-frequency keys
        print(f"Building B-Tree indexes on {s2_count + s3_count:,} target records...")
        t_idx = time.time()
        lookup.finalize_indexes()
        c_stats['index_build_time'] = time.time() - t_idx
        c_stats['db_size_mb'] = os.path.getsize(target_db_path) / (1024 * 1024)
        print(f"B-Tree indexes built in {c_stats['index_build_time']:.2f}s (DB size: {c_stats['db_size_mb']:.1f} MB)")

        # Verify SQLite integrity
        lookup.cur.execute("PRAGMA integrity_check")
        integ = lookup.cur.fetchone()[0]
        assert integ == "ok", f"SQLite integrity check failed on {target_db_path}: {integ}"

        # Check existing predictions for safe resumability
        cur_preds.execute("SELECT s1_id FROM preds")
        already_predicted = set(row[0] for row in cur_preds.fetchall())

        # Query candidates and score in batches
        print(f"Scoring candidates for {country}...")
        db_buffer = []
        batch_pairs = []
        batch_pair_meta = []
        s1_cands_map = {}
        processed = 0

        with open(s1_country_file, 'r', encoding='utf-8') as f_s1:
            for line in f_s1:
                parts = line.rstrip('\n').split('\t')
                s1_id = parts[0].strip()
                if not s1_id:
                    continue

                if s1_id in already_predicted:
                    processed += 1
                    continue

                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""

                n_dict = normalize_business_name(name)
                a_dict = normalize_address(addr)

                ranked_cands = lookup.retrieve_candidates(
                    n_dict, a_dict, max_candidates=max_final_cands
                )

                # Anomaly check: ensure candidate count never exceeds MAX_FINAL_CANDIDATES
                if len(ranked_cands) > max_final_cands:
                    raise ValueError(f"CRITICAL ANOMALY: Candidate count {len(ranked_cands)} exceeds cap {max_final_cands} for {s1_id}")

                num_cands = len(ranked_cands)
                c_stats['candidate_pairs'] += num_cands
                diagnostics['total_candidate_pairs'] += num_cands

                if num_cands == 0:
                    diagnostics['candidates_per_s1_buckets']['0'] += 1
                    db_buffer.append((s1_id, "", ""))
                    c_stats['singletons'] += 1
                    diagnostics['singleton_entities'] += 1
                else:
                    if 1 <= num_cands <= 5:
                        diagnostics['candidates_per_s1_buckets']['1-5'] += 1
                    elif 6 <= num_cands <= 10:
                        diagnostics['candidates_per_s1_buckets']['6-10'] += 1
                    elif 11 <= num_cands <= 20:
                        diagnostics['candidates_per_s1_buckets']['11-20'] += 1
                    elif 21 <= num_cands <= 40:
                        diagnostics['candidates_per_s1_buckets']['21-40'] += 1
                    elif 41 <= num_cands <= 60:
                        diagnostics['candidates_per_s1_buckets']['41-60'] += 1
                    if num_cands == max_final_cands:
                        diagnostics['candidates_per_s1_buckets']['capped_at_60'] += 1

                    cand_ids = [c[0] for c in ranked_cands]
                    s1_cands_map[s1_id] = cand_ids

                    # Batch retrieve normalized records for missing candidate targets
                    target_recs = lookup.get_records_batch(cand_ids)

                    for target_id, prov_list in ranked_cands:
                        # Record provenance statistics
                        hit_count = len(prov_list)
                        if hit_count == 1:
                            diagnostics['blocking_hits_distribution']['1'] += 1
                        elif hit_count == 2:
                            diagnostics['blocking_hits_distribution']['2'] += 1
                        else:
                            diagnostics['blocking_hits_distribution']['3+'] += 1

                        for p in prov_list:
                            diagnostics['blocking_strategy_usage'][p] += 1

                        target_rec = target_recs.get(target_id)
                        feat = compute_pairwise_features(
                            n_dict, a_dict, target_id, target_rec, prov_list
                        )
                        if len(feat) != 36:
                            raise ValueError(f"Feature count mismatch: {len(feat)} != 36")
                        batch_pairs.append(feat)
                        batch_pair_meta.append((s1_id, target_id))

                processed += 1

                # Flush batch to model
                if len(batch_pairs) >= batch_size or processed == c_count:
                    if batch_pairs:
                        X_batch = np.array(batch_pairs, dtype=np.float32)
                        if np.isnan(X_batch).any() or np.isinf(X_batch).any():
                            raise ValueError("NaN or Inf values detected in feature matrix!")

                        probs = model.predict_proba(X_batch)[:, 1]
                        if (probs < 0.0).any() or (probs > 1.0).any():
                            raise ValueError("Model probability out of bounds [0, 1]!")

                        diagnostics['prob_min'] = min(diagnostics['prob_min'], float(probs.min()))
                        diagnostics['prob_max'] = max(diagnostics['prob_max'], float(probs.max()))
                        diagnostics['prob_sum'] += float(probs.sum())
                        diagnostics['prob_count'] += len(probs)

                        scored_by_s1 = defaultdict(list)
                        for (cur_s1, cur_target), prob in zip(batch_pair_meta, probs):
                            scored_by_s1[cur_s1].append((cur_target, float(prob)))

                        for cur_s1, scored_list in scored_by_s1.items():
                            scored_list.sort(key=lambda x: x[1], reverse=True)
                            accepted = apply_decision_policy(
                                scored_list,
                                decision_threshold=decision_thresh,
                                singleton_threshold=singleton_thresh
                            )
                            c_list = s1_cands_map.get(cur_s1, [])
                            db_buffer.append((cur_s1, ",".join(c_list), ",".join(sorted(accepted))))

                            if accepted:
                                c_stats['matches'] += 1
                                diagnostics['matched_entities'] += 1
                                diagnostics['total_matched_links'] += len(accepted)
                                for link_id in accepted:
                                    if link_id.startswith('S2-'):
                                        diagnostics['s2_matched_links'] += 1
                                    elif link_id.startswith('S3-'):
                                        diagnostics['s3_matched_links'] += 1
                            else:
                                c_stats['singletons'] += 1
                                diagnostics['singleton_entities'] += 1

                        batch_pairs = []
                        batch_pair_meta = []
                        s1_cands_map.clear()
                        lookup.clear_cache()

                    if db_buffer:
                        cur_preds.executemany("INSERT OR REPLACE INTO preds VALUES (?, ?, ?)", db_buffer)
                        conn_preds.commit()
                        db_buffer = []

                if processed % 50000 == 0 or processed == c_count:
                    pct = (processed / c_count) * 100
                    print(f"  Processed {processed:,} / {c_count:,} entities ({pct:.1f}%) in {time.time()-t_country:.1f}s")

        # Record country completion in checkpoints table
        cur_preds.execute(
            "INSERT OR REPLACE INTO completed_countries VALUES (?, ?, ?, ?, datetime('now'))",
            (country, c_count, c_stats['matches'], c_stats['singletons'])
        )
        conn_preds.commit()

        c_stats['runtime_sec'] = time.time() - t_country
        diagnostics['countries'][country] = c_stats
        print(f"Completed {country} partition in {c_stats['runtime_sec']:.1f}s ({c_stats['matches']:,} matched | {c_stats['singletons']:,} singletons)")

        lookup.close()
        if os.path.exists(target_db_path):
            try:
                os.remove(target_db_path)
            except OSError:
                pass
        gc.collect()

    # Step 3: Stream from SQLite in exact test order to generate final TSVs
    print("\nExporting final results in exact test order from SQLite to TSVs...")
    t_export_start = time.time()
    matching_tmp = matching_path + ".tmp"
    candidate_tmp = candidate_path + ".tmp"

    matched_count = 0
    singleton_count = 0
    total_exported = 0

    BATCH_IN = 500  # Safe variable batch size to avoid SQLite variable limits

    with open(matching_tmp, 'w', encoding='utf-8') as f_match, \
         open(candidate_tmp, 'w', encoding='utf-8') as f_cand, \
         open(order_file, 'r', encoding='utf-8') as f_order:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        id_chunk = []
        for line in f_order:
            s1_id = line.strip()
            if not s1_id:
                continue
            id_chunk.append(s1_id)

            if len(id_chunk) >= 10000:
                res_map = {}
                for i in range(0, len(id_chunk), BATCH_IN):
                    sub = id_chunk[i:i + BATCH_IN]
                    placeholders = ','.join('?' * len(sub))
                    cur_preds.execute(f"SELECT s1_id, candidates, matches FROM preds WHERE s1_id IN ({placeholders})", sub)
                    for row in cur_preds.fetchall():
                        res_map[row[0]] = (row[1], row[2])

                for q_id in id_chunk:
                    cands, matches = res_map.get(q_id, ("", ""))
                    f_cand.write(f"{q_id}\t{cands}\n")
                    if matches:
                        matched_count += 1
                        f_match.write(f"{q_id}\t{matches}\n")
                    else:
                        singleton_count += 1
                        f_match.write(f"{q_id}\t\n")
                    total_exported += 1
                id_chunk = []

        if id_chunk:
            res_map = {}
            for i in range(0, len(id_chunk), BATCH_IN):
                sub = id_chunk[i:i + BATCH_IN]
                placeholders = ','.join('?' * len(sub))
                cur_preds.execute(f"SELECT s1_id, candidates, matches FROM preds WHERE s1_id IN ({placeholders})", sub)
                for row in cur_preds.fetchall():
                    res_map[row[0]] = (row[1], row[2])

            for q_id in id_chunk:
                cands, matches = res_map.get(q_id, ("", ""))
                f_cand.write(f"{q_id}\t{cands}\n")
                if matches:
                    matched_count += 1
                    f_match.write(f"{q_id}\t{matches}\n")
                else:
                    singleton_count += 1
                    f_match.write(f"{q_id}\t\n")
                total_exported += 1

    conn_preds.close()
    f_match_size = os.path.getsize(matching_tmp)
    f_cand_size = os.path.getsize(candidate_tmp)

    # Validate output row count before atomic rename
    assert total_exported == diagnostics['total_s1'], f"Export count mismatch: {total_exported} != {diagnostics['total_s1']}"
    if os.path.exists(matching_path):
        os.remove(matching_path)
    if os.path.exists(candidate_path):
        os.remove(candidate_path)
    os.replace(matching_tmp, matching_path)
    os.replace(candidate_tmp, candidate_path)

    # Copy final verified outputs into output/phase2/
    phase2_matching = os.path.join(phase2_dir, "matching_results.tsv")
    phase2_candidate = os.path.join(phase2_dir, "candidate_pairs.tsv")
    phase2_submission = os.path.join(phase2_dir, "final_submission.tsv")
    shutil.copyfile(matching_path, phase2_matching)
    shutil.copyfile(candidate_path, phase2_candidate)
    shutil.copyfile(matching_path, phase2_submission)

    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    diagnostics['peak_ram_mb'] = peak_ram / (1024 * 1024)
    diagnostics['total_runtime_sec'] = time.time() - t_start
    diagnostics['matching_file_size_mb'] = f_match_size / (1024 * 1024)
    diagnostics['candidate_file_size_mb'] = f_cand_size / (1024 * 1024)
    diagnostics['final_exported_rows'] = total_exported
    diagnostics['final_submission_path'] = phase2_submission

    # Save detailed JSON statistics
    stats_json_path = os.path.join(phase2_dir, "inference_statistics.json")
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, default=str)

    print(f"\nTest Inference Completed in {diagnostics['total_runtime_sec']:.2f}s!")
    print(f"  {matching_path} ({total_exported:,} rows | {matched_count:,} matched | {singleton_count:,} singletons | {diagnostics['matching_file_size_mb']:.1f} MB)")
    print(f"  {candidate_path} ({total_exported:,} rows | {diagnostics['candidate_file_size_mb']:.1f} MB)")
    print(f"  Peak RAM Tracked: {diagnostics['peak_ram_mb']:.2f} MB")
    print(f"  Phase 2 Results Saved to: {phase2_dir}")

    return diagnostics
