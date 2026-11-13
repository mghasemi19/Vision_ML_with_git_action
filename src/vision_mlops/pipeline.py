"""Training/evaluation entrypoint and immutable artifact metadata."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from vision_mlops.data import load_dataset, split_data
from vision_mlops.model import fit_models, metrics_for, quality_failures


def run_pipeline(output_dir, check_quality=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels = load_dataset()
    train_idx, val_idx, test_idx = split_data(images, labels)
    if (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx)
            or set(val_idx) & set(test_idx)):
        raise RuntimeError("Data splits overlap")
    candidate, reference = fit_models(images, labels, train_idx)
    val = metrics_for(candidate, images[val_idx], labels[val_idx])
    # Evaluate test exactly once for this fixed training specification.
    candidate_test = metrics_for(candidate, images[test_idx], labels[test_idx])
    reference_test = metrics_for(reference, images[test_idx], labels[test_idx])
    failures = quality_failures(candidate_test, reference_test)
    dataset_hash = hashlib.sha256(images.tobytes() + labels.tobytes()).hexdigest()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "sklearn.load_digits (8x8 grayscale images)",
        "dataset_sha256": dataset_hash,
        "seed": 42,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "validation": val,
        "test": candidate_test,
        "naive_reference_test": reference_test,
        "quality_passed": not failures,
        "quality_failures": failures,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "warning": "Educational handwritten-digit classifier; not a production incumbent comparison.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    # Only export the candidate model when the quality criteria pass.
    if not failures:
        joblib.dump(candidate, output_dir / "model.joblib")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--allow-fail", action="store_true", help="Diagnostic only; does not bypass artifact gate")
    args = parser.parse_args()
    report = run_pipeline(args.output_dir)
    print(json.dumps({"test": report["test"], "quality_passed": report["quality_passed"], "quality_failures": report["quality_failures"]}, indent=2))
    if not report["quality_passed"] and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
