import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import time
from intervaltree import Interval, IntervalTree

def has_any_overlap(row, df2, start_col='start', end_col='end'):
    s1, e1 = row[start_col], row[end_col]
    if df2.empty:
        return False
    starts = df2[start_col].to_numpy()
    ends = df2[end_col].to_numpy()
    return np.any(np.maximum(s1, starts) <= np.minimum(e1, ends))

def prebuild_reference_trees(reference, start_col='start', end_col='end'):
    ref_by_file = dict(tuple(reference.groupby('filename')))
    ref_trees = {fn: build_interval_tree(df, start_col, end_col) for fn, df in ref_by_file.items()}
    return ref_by_file, ref_trees

def build_interval_tree(df, start_col='start', end_col='end'):
    tree = IntervalTree()
    for row in df.itertuples(index=False):
        start = getattr(row, start_col)
        end = getattr(row, end_col)
        if start < end:  # adding valid intervals
            tree.add(Interval(start, end))
        else:
            continue  
    return tree

def get_continuous_results(evaluation, reference,
                           ref_by_file=None, ref_trees=None):
    tp = fp = fn = 0

    if ref_by_file is None or ref_trees is None:
        ref_by_file = dict(tuple(reference.groupby('filename')))
        ref_trees = {fn: build_interval_tree(df) for fn, df in ref_by_file.items()}

    eval_by_file = dict(tuple(evaluation.groupby('filename')))
    eval_trees = {fn: build_interval_tree(df) for fn, df in eval_by_file.items()}

    for filename, ref_tree in ref_trees.items():
        eval_tree = eval_trees.get(filename, IntervalTree())
        for row in ref_by_file[filename].itertuples(index=False):
            if eval_tree.overlaps(row.start, row.end):
                tp += 1
            else:
                fn += 1

    for filename, eval_tree in eval_trees.items():
        ref_tree = ref_trees.get(filename, IntervalTree())
        for row in eval_by_file[filename].itertuples(index=False):
            if not ref_tree.overlaps(row.start, row.end):
                fp += 1

    return {'TP': tp, 'FP': fp, 'FN': fn}

def compute_custom_pr_curve(evaluation, reference, thresholds):
    precision_list = []
    recall_list = []
    f1_list = []


    # Pre-sort evaluation by descending score
    evaluation = evaluation.sort_values(by='score', ascending=False).reset_index(drop=True)

    ref_by_file, ref_trees = prebuild_reference_trees(reference)

    for threshold in tqdm(thresholds, desc="Computing PR curve"):
        filtered_eval = evaluation[evaluation['score'] >= threshold]
        results = get_continuous_results(filtered_eval, reference,
                                         ref_by_file=ref_by_file, ref_trees=ref_trees)

        tp, fp, fn = results['TP'], results['FP'], results['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list, thresholds

# def get_continuous_results(evaluation, reference):
#     """
#     Evaluate the True Positives (TP), False Positives (FP), and False Negatives (FN) for each unique class label
#     in the ground truth DataFrame based on the evaluation DataFrame.

#     This function is suitable for long, continuous segments of data and is designed to compare two sets of 
#     time-stamped annotations, which could either be predictions and ground truths, or two different sets of annotations.
    
#     Args:
#         evaluation: pd.DataFrame
#             DataFrame containing the evaluation results, which could be detection scores or another set of annotations.
#             The DataFrame should have columns like 'label', 'start', 'end', etc.
#         reference: pd.DataFrame
#             DataFrame containing the ground truth labels, with similar structure to `predicted`.

#     Returns: dict
#         A dictionary where keys are the unique class labels, and the values are dictionaries containing the counts of TP, FP, FN.
#     """
#     start_time = time.time()
#     tp = fp = fn = 0   

#     # Group by filename for efficiency
#     ref_by_file = dict(tuple(reference.groupby('filename')))
#     eval_by_file = dict(tuple(evaluation.groupby('filename')))
    

#     # We are now going to calculate the TP and FN. Note that this calculation covers instances where we have one prediction that overlaps with
#     # multiple ground truths. In this case, the TP will be incremented for each ground truth
#     # We are also considering Multiple preditcions for one ground truth, in this case, only one of the predictions will increment the TP.
#     # Another assumption made here is that even if only 1% of the prediction falls within the ground truth it will count as a TP

#     # Compute TPs and FNs
#     for filename, ref_group in ref_by_file.items():
#         eval_group = eval_by_file.get(filename, pd.DataFrame(columns=evaluation.columns))
#         for _, gt_row in ref_group.iterrows():
#             if has_any_overlap(gt_row, eval_group):
#                 tp += 1
#             else:
#                 fn += 1
                
    
#     # Calculate FP:
#     # Loop through each predicted filtered entry
#     # False Positives
#     # Compute FPs
#     for filename, eval_group in eval_by_file.items():
#         ref_group = ref_by_file.get(filename, pd.DataFrame(columns=reference.columns))
#         for _, pred_row in eval_group.iterrows():
#             if not has_any_overlap(pred_row, ref_group):
#                 fp += 1
#     end_time = time.time()
#     # print(f"Execution time: {end_time - start_time:.2f} seconds")
#     return {
#         'TP': tp,
#         'FP': fp,
#         'FN': fn
#     }


def calculate_metrics(TP, FP, FN, TN=None, total_time_units=None):
    """
    Calculate classification metrics like precision, recall, F1-score, 
    and optionally accuracy and False Positive Rate (FPR) per time unit.

    Args:
        TP: int 
            Number of True Positives
        FP: int
            Number of False Positives
        FN: int
            Number of False Negatives
        total_time_units: float
            The total duration in an arbitrary unit of time (e.g., hours, minutes). 
            If provided, will calculate FPR per time unit.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Initialize a dictionary to store the metrics
    metrics = {}

    # Calculate precision
    if TP + FP == 0:
        metrics['Precision'] = 0
    else:
        metrics['Precision'] = TP / (TP + FP)

    # Calculate recall
    if TP + FN == 0:
        metrics['Recall'] = 0
    else:
        metrics['Recall'] = TP / (TP + FN)

    # Calculate F1-score
    if metrics['Precision'] + metrics['Recall'] == 0:
        metrics['F1-Score'] = 0
    else:
        metrics['F1-Score'] = 2 * metrics['Precision'] * metrics['Recall'] / (metrics['Precision'] + metrics['Recall'])

    # Calculate accuracy
    if TN is not None:
        if TP + FP + TN + FN == 0:
            metrics['Accuracy'] = 0
        else:
            metrics['Accuracy'] = (TP + TN) / (TP + FP + TN + FN)

    # Calculate FPR per hour if total_time_units is provided
    if total_time_units is not None:
        if total_time_units == 0:
            metrics['FPR_per_time_unit'] = 0
        else:
            metrics['FPR_per_time_unit'] = FP / total_time_units


    return metrics

def evaluate_thresholded(evaluation, reference, threshold_min=0, threshold_max=1, threshold_inc=0.05, total_time_units=None, output_folder=None, audio_representation=None):
    """
    Evaluate the performance of sound detections for both shorter clips and longer continuous files.
    This function iterates over a range of detection score thresholds to evaluate performance metrics 
    like True Positive, True Negative, False Positive, and False Negative counts for each class.
    Depending on the 'type' specified, additional metrics may be calculated. 
    It also calculates macro and micro averages for metrics like Precision, Recall, and F1-Score.

    Args:
        evaluation: str
            Path to the CSV file containing evaluation results, which could be detection scores or a set of annotations.
        reference: str 
            Path to the CSV file containing ground-truth reference.
        threshold_min: float
            Minimum threshold for detection. Defaults to 0.
        threshold_max: float
            Maximum threshold for detection. Defaults to 1.
        threshold_inc: float
            Threshold increment for each step. Defaults to 0.05.
        total_time_units: float or None
            The total duration in arbitrary time units over which the detections were made (e.g., hours, minutes).
            This is necessary to calculate the False Positive Rate per unit time. Defaults to None.
        output_folder: str or None
            The folder where to save the output CSV files. If None, the current directory is used.
            Defaults to None.
    Returns:
        tuple: Two Pandas DataFrames.
            - The first DataFrame contains detection results for each threshold and class.
            - The second DataFrame contains classification metrics for each threshold and class, also including macro and micro averages.
    """
    # Read the detection and annotation files
    evaluation = pd.read_csv(evaluation)
    reference = pd.read_csv(reference)

    # Determine the output folder
    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    # List of metrics to calculate
    metrics_to_calculate = ['Precision', 'Recall', 'F1-Score'] # These metrics will always be calculated no matter the type

    if total_time_units is not None:
        metrics_to_calculate.append('FPR_per_time_unit')

    if threshold_min == threshold_max:
        thresholds = [threshold_min]
    else:
        all_thresholds = sorted(evaluation['score'].unique(), reverse=True)
        # thresholds = all_thresholds[:-1]  # Drop the last one
        # thresholds = list(np.arange(threshold_min, threshold_max, threshold_inc))
        # if thresholds[-1] != threshold_max: # Manually including the last threshold
        #     thresholds.append(threshold_max)
    
    for threshold in thresholds:
        threshold = round(threshold, 5) # rounding the threshold to avoid floating-point issues
        
        # Get results
        result = get_continuous_results(evaluation, reference)
        
        TP = result['TP']
        FP = result['FP']
        FN = result['FN']
        TN = None  # Not defined in this context
        
        metrics = calculate_metrics(TP, FP, FN, TN, total_time_units=total_time_units)


    # Convert to DataFrames
    results_df = pd.DataFrame.from_dict(result, orient='index')
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

    # Save to CSV
    # results_df.to_csv(output_folder / 'results.csv')
    metrics_df.to_csv(output_folder / 'metrics.csv')

    print(f"All results and metrics saved to {output_folder}")
    return results_df, metrics_df


def evaluate_score_based(evaluation, reference, output_folder=None):
    """
    Evaluate the performance of sound detections based on scores using a custom segment-aware PR curve.
    
    Args:
        evaluation_path: str
            Path to the CSV file containing evaluation results (must include 'score', 'start', 'end', etc.).
        reference_path: str 
            Path to the CSV file containing ground-truth reference annotations.
        output_folder: str or None
            Folder where to save plots. If None, plots are not saved.
    
    Returns:
        Tuple of (precision_list, recall_list, thresholds)
    """
    # Load data
    evaluation_df = pd.read_csv(evaluation)
    reference_df = pd.read_csv(reference)

    # Generate thresholds and compute PR curve
    # thresholds = np.linspace(1.0, 0.0, 100, endpoint=False)
    thresholds = sorted(evaluation_df['score'].unique(), reverse=True)
    # all_thresholds = sorted(np.round(evaluation_df['score'].unique(), 3), reverse=True)
    # thresholds = np.unique(all_thresholds)  # Ensure uniqueness after rounding
    # thresholds = all_thresholds[:-1]  # Drop the last one
    
    precision_list, recall_list, f1_list, thresholds = compute_custom_pr_curve(evaluation_df, reference_df, thresholds)

    from sklearn.metrics import auc
    
    # Compute Average Precision (AUC)
    recall_arr = np.array(recall_list)
    precision_arr = np.array(precision_list)
    sorted_indices = np.argsort(recall_arr)
    recall_sorted = recall_arr[sorted_indices]
    precision_sorted = precision_arr[sorted_indices]
    average_precision = auc(recall_sorted, precision_sorted)
    print(f"Average Precision (AUC): {average_precision:.4f}")

    # Compute Average Precision (optional)
    # average_precision = auc(recall_list, precision_list)

    # Plot PR Curve
    if output_folder is not None:
        output_folder = Path(output_folder).resolve()
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save PR plot
        plt.figure()
        plt.plot(recall_list, precision_list)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        # plt.title(f"Custom PR Curve (AP = {average_precision:.2f})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(0.35, 1.05)
        plt.grid(True)
        plt.savefig(output_folder / "pr_curve.png")
        plt.close()

        print(f"PR Curve saved to {output_folder / 'pr_curve.png'}")
    
        plt.figure()
        plt.plot(thresholds, f1_list)
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Threshold")
        plt.grid(True)
        plt.xlim(-0.05, 1.05)
        plt.ylim(0.35, 1.05)
        plt.savefig(output_folder / "f1_vs_threshold.png")
        plt.close()

        print(f"F1 vs Threshold plot saved to {output_folder / 'f1_vs_threshold.png'}")

def main():
    import argparse
        
    parser = argparse.ArgumentParser(description="Evaluate the performance of sound detection.")
    parser.add_argument('reference', type=str, help='Path to the .csv file containing ground truth reference.')
    parser.add_argument('--evaluation', type=str, default=None, help='Path to the .csv file containing evaluation results, which may include detection scores or annotations.')
    parser.add_argument('--model_dirs', nargs='*', default=None,
    help='Used in "compare_auc" mode. List of folders with detections_raw.csv to compare.')
    parser.add_argument('--model_labels', nargs='*', default=None,
    help='List of custom labels for each model to show in the legend. Must match the order and length of --model_dirs.')
    parser.add_argument('--mode', choices=['thresholded', 'score_based', 'compare_auc'], default='thresholded', help='Type of evaluation to perform: "thresholded" for threshold-based evaluation or "score_based" for score-based evaluation.')
    parser.add_argument('--threshold_min', default=0, type=float, help='Minimum threshold for detection.')
    parser.add_argument('--threshold_max', default=1, type=float, help='Maximum threshold for detection.')
    parser.add_argument('--threshold_inc', default=0.05, type=float, help='Threshold increment for each step.')
    parser.add_argument('--total_time_units', default=None, type=float, help='The total duration in arbitrary time units over which the detections were made (e.g., hours, minutes).' \
            ' This is necessary to calculate the False Positive Rate per unit time. Defaults to None.')
    parser.add_argument('--output_folder', default=None, type=str, help='Location to output the performance results. For instance: metrics/')
    parser.add_argument('--audio_representation', default=None, type=str, help='Path to audio representation config file.')
    
    args = parser.parse_args()
    if args.mode == 'thresholded':
        results, metrics = evaluate_thresholded(
            evaluation=args.evaluation,
            reference=args.reference,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_inc=args.threshold_inc,
            total_time_units=args.total_time_units,
            output_folder=args.output_folder,
            audio_representation=args.audio_representation
        )
    elif args.mode == 'score_based':
        evaluate_score_based(
            evaluation=args.evaluation,
            reference=args.reference,
            output_folder=args.output_folder
        )
    elif args.mode == 'compare_auc':
        if not args.model_dirs:
            raise ValueError("In 'compare_auc' mode, --model_dirs must be provided.")
        compare_auc_models(args.model_dirs, args.reference, args.output_folder, model_labels=args.model_labels)
    else:
        raise ValueError("Invalid mode. Choose either 'thresholded' or 'score_based'.")

if __name__ == "__main__":
    main()
