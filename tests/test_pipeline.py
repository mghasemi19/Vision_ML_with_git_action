import json
from vision_mlops.pipeline import run_pipeline


def test_pipeline_exports_report_and_model(tmp_path):
    report = run_pipeline(tmp_path)
    assert report["quality_passed"]
    assert report["split_sizes"]["train"] + report["split_sizes"]["validation"] + report["split_sizes"]["test"] == 1797
    assert (tmp_path / "model.joblib").is_file()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["dataset_sha256"] == report["dataset_sha256"]
