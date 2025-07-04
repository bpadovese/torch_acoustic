import subprocess
import sys
from pathlib import Path

def run_metrics(subdir):
    # Define base paths
    base_results_dir = Path("results/") / subdir
    annotations_path = "common/annotations/RB_annotations.csv"

    # Ensure output directory exists
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Run thresholded mode
    thresholded_cmd = [
        "python", "metrics.py",
        annotations_path,
        "--evaluation", str(base_results_dir / "detections.csv"),
        "--threshold_min", "0.5",
        "--threshold_max", "0.5",
        "--mode", "thresholded",
        "--output_folder", str(base_results_dir)
    ]

    # Run score_based mode
    score_based_cmd = [
        "python", "metrics.py",
        annotations_path,
        "--evaluation", str(base_results_dir / "detections_raw.csv"),
        "--mode", "score_based",
        "--output_folder", str(base_results_dir)
    ]

    print("\n[Running Thresholded Evaluation]")
    subprocess.run(thresholded_cmd, check=True)

    print("\n[Running Score-Based Evaluation]")
    subprocess.run(score_based_cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_metrics_pair.py <subdir_name>")
        sys.exit(1)
    
    subdir = sys.argv[1]
    run_metrics(subdir)
