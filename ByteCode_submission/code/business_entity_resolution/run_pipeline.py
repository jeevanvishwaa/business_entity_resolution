"""
Master CLI Entry Point for Amazon ML Challenge 2026: Business Entity Resolution.
Memory-optimized for 12GB RAM systems.
"""

import os
import sys
import io
import time
import argparse
import subprocess
import gc
import joblib
import json
import pandas as pd
import numpy as np

# Ensure module path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Safe UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.config import Config
from src.dataset_audit import run_audit
from src.pipeline import prepare_training_and_val_pairs
from src.train import train_and_evaluate
from src.predict import run_test_inference

def log_experiment(exp_id, model_name, config_summary, metrics):
    os.makedirs(Config.EXPERIMENTS_DIR, exist_ok=True)
    res_csv = os.path.join(Config.EXPERIMENTS_DIR, "results.csv")
    row = {
        'experiment_id': exp_id,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_name,
        'config': config_summary,
        'macro_f05': metrics.get('macro_f05', 0.0),
        'macro_precision': metrics.get('macro_precision', 0.0),
        'macro_recall': metrics.get('macro_recall', 0.0),
        'singleton_accuracy': metrics.get('singleton_accuracy', 0.0),
        'decision_threshold': metrics.get('decision_threshold', 0.65),
        'singleton_threshold': metrics.get('singleton_threshold', 0.55)
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(res_csv):
        df_row.to_csv(res_csv, mode='a', header=False, index=False)
    else:
        df_row.to_csv(res_csv, index=False)

    log_md = os.path.join(Config.BASE_DIR, "EXPERIMENT_LOG.md")
    with open(log_md, 'a', encoding='utf-8') as f:
        f.write(f"\n### Experiment `{exp_id}` - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Model:** {model_name}\n")
        f.write(f"- **Config:** {config_summary}\n")
        f.write(f"- **Macro F_0.5:** {row['macro_f05']:.4f}\n")
        f.write(f"- **Precision:** {row['macro_precision']:.4f} | **Recall:** {row['macro_recall']:.4f}\n")
        f.write(f"- **Singleton Accuracy:** {row['singleton_accuracy']:.4f}\n")
        f.write(f"- **Thresholds:** decision={row['decision_threshold']}, singleton={row['singleton_threshold']}\n")

def run_submission_validation():
    print("\n==========================================")
    print("RUNNING OFFICIAL SUBMISSION VALIDATION")
    print("==========================================")
    cmd = [
        sys.executable,
        Config.VALIDATE_SCRIPT,
        "--matching", Config.MATCHING_RESULTS_TSV,
        "--candidate", Config.CANDIDATE_PAIRS_TSV,
        "--test-dir", os.path.join(Config.DATA_DIR, "test")
    ]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument("--mode", default="all", choices=[
        'audit', 'train', 'predict', 'validate_submission', 'all'
    ])
    parser.add_argument("--sample-size", type=int, default=15000,
                        help="S1 training entity sample size (default 15000 for 12GB RAM)")
    args = parser.parse_args()

    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)

    model_path = os.path.join(Config.MODELS_DIR, "final_model.joblib")
    config_path = os.path.join(Config.MODELS_DIR, "final_threshold_config.json")

    # === AUDIT ===
    if args.mode in ('audit', 'all'):
        run_audit(data_dir=Config.DATA_DIR, reports_dir=Config.REPORTS_DIR)
        gc.collect()

    # === TRAIN ===
    model = None
    best_config = None

    if args.mode in ('train', 'all'):
        X_train, y_train, val_pairs_by_s1, val_gt = prepare_training_and_val_pairs(
            sample_s1_size=args.sample_size
        )

        # LightGBM (primary model)
        model, best_config = train_and_evaluate(
            X_train, y_train, val_pairs_by_s1, val_gt,
            model_type='lightgbm', reports_dir=Config.REPORTS_DIR
        )
        log_experiment('EXP-001-LGBM', 'LightGBM',
                       f'sample={args.sample_size}, 300 trees, lr=0.08, 36 features',
                       best_config)

        # Save model
        joblib.dump(model, model_path)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2)
        print(f"Saved model to {model_path}")

        # Free training data
        del X_train, y_train, val_pairs_by_s1, val_gt
        gc.collect()

    # === PREDICT ===
    if args.mode in ('predict', 'all'):
        if model is None:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                model = joblib.load(model_path)
                with open(config_path, 'r') as f:
                    best_config = json.load(f)
            else:
                print("ERROR: No trained model found. Run --mode train first.")
                return

        run_test_inference(model, best_config,
                           test_dir=os.path.join(Config.DATA_DIR, "test"),
                           output_dir=Config.OUTPUT_DIR)

    # === VALIDATE ===
    if args.mode in ('validate_submission', 'all'):
        code = run_submission_validation()
        if code == 0:
            print("\n*** VALIDATION: PASS ***")
        else:
            print(f"\n*** VALIDATION: FAILED (exit {code}) ***")

if __name__ == '__main__':
    main()
