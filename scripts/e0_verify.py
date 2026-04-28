"""E0: Pipeline verification script for pi0.5 on Libero.
Verifies that the model/dataset pipeline works by loading a few batches.
"""
import json
import os
import sys

def main():
    output_dir = os.environ.get("EVAL_OUTPUT_DIR", "/mnt/default_storage/qiming/openpi/eval_results")
    os.makedirs(output_dir, exist_ok=True)

    import openpi.training.config as cfg
    import openpi.training.data_loader as dl

    c = cfg.get_config("pi05_libero")
    loader = dl.create_data_loader(c, shuffle=False, num_batches=5, framework="pytorch")

    count = 0
    for obs, act in loader:
        count += 1
        print(f"Batch {count}: act shape={act.shape}")
        if count >= 3:
            break

    print(f"E0 pipeline verification passed: {count} batches loaded.")
    result = {
        "experiment_name": "E0_raw_base",
        "notes": f"Pipeline verified: loaded {count} batches",
    }
    with open(os.path.join(output_dir, "raw_base_offline_eval.json"), "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
