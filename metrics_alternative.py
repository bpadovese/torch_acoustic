from matplotlib import pyplot as plt

def find_overlap(row, df2, filename_col='filename', start_col='start', end_col='end'):
    """
    Check if a given row overlaps temporally (time) with any row in another DataFrame (df2).

    Args:
        row: pd.Series
            A single row (pandas Series) from a DataFrame that contains the 'start' and 'end' columns,
            as well as a 'filename' column that specifies the audio file to which the row belongs.   
        df2: pd.DataFrame
            A DataFrame containing rows with 'start' and 'end' columns, as well as a 'filename' column.
            This DataFrame is checked for overlap against the given row.
        filename_col: str
            The name of the column in both dataframes that contains the filename. 
            Defaults to 'filename'.
        start_col: str
            The name of the column in both dataframes that contains the start time. 
            Defaults to 'start'.
        end_col: str
            The name of the column in both dataframes that contains the end time. 
            Defaults to 'end'.

    Returns: bool
        Returns True if the given row overlaps with any row in df2 for the same filename,
        otherwise returns False.

    Example:

    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ... 'filename': ['file1', 'file2'],
    ... 'start': [1, 3],
    ... 'end': [2, 4]
    ... })
    >>> df2 = pd.DataFrame({
    ... 'filename': ['file1', 'file1', 'file2'],
    ... 'start': [1.5, 2.5, 3.5],
    ... 'end': [2.5, 3.5, 4.5]
    ... })
    >>> row = df1.iloc[0]
    >>> find_overlap(row, df2)
    True
    
    >>> row = df1.iloc[1]
    >>> find_overlap(row, df2)
    True
    """

    interval1 = pd.Interval(row[start_col], row[end_col])
    
    matching_df2 = df2[df2[filename_col] == row[filename_col]]
    
    for _, row2 in matching_df2.iterrows():
        interval2 = pd.Interval(row2[start_col], row2[end_col])
        if interval1.overlaps(interval2):
            return True
                
    return False

def plot_roc_curve(fpr, tpr, auc, output_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(precision, recall, ap, output_path):
    # Drop misleading (precision=0, recall=0) edge points
    while len(precision) > 1 and precision[0] == 0 and recall[0] == 0:
        precision = precision[1:]
        recall = recall[1:]

    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def compute_auc_and_pr(evaluation_df, reference_df):
    # Sort once (for PR behavior); okay for AUC too
    evaluation_df_sorted = evaluation_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # Match once
    y_true, y_score = prepare_binary_classification_targets(evaluation_df_sorted, reference_df)
    
    # Compute AUC
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

    # Compute PR
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'average_precision': ap
    }

def prepare_binary_classification_targets(evaluation_df, reference_df):
    y_true = []
    y_score = []

    matched_preds = set()
    matched_refs = set()

    for pred_idx, pred_row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df), desc="Greedy matching"):
        pred_file = pred_row['filename']
        pred_start, pred_end = pred_row['start'], pred_row['end']
        pred_score = pred_row['score']

        matching_refs = reference_df[reference_df['filename'] == pred_file]

        best_match = None
        for ref_idx, ref_row in matching_refs.iterrows():
            if ref_idx in matched_refs:
                continue
            if intervals_overlap(pred_start, pred_end, ref_row['start'], ref_row['end']):
                best_match = ref_idx
                break

        if best_match is not None:
            matched_preds.add(pred_idx)
            matched_refs.add(best_match)
            y_true.append(1)
            y_score.append(pred_score)
        else:
            # Check if this prediction overlaps with a ref that's already matched
            overlaps_with_matched = any(
                intervals_overlap(pred_start, pred_end, reference_df.loc[ref_idx, 'start'], reference_df.loc[ref_idx, 'end'])
                for ref_idx in matched_refs
                if reference_df.loc[ref_idx, 'filename'] == pred_file
            )
            
            if not overlaps_with_matched:
                # Count only if it doesn't match any already-used reference
                y_true.append(0)
                y_score.append(pred_score)
            # Else, ignore the prediction altogether

    # Unmatched refs are FNs, don't include in y_score
    print(f"Unmatched refs (FN): {len(reference_df) - len(matched_refs)}")

    from collections import Counter
    print("y_true counts:", Counter(y_true))
    print("y_score min/max:", min(y_score), max(y_score))

    return y_true, y_score


# def evaluate_score_based(evaluation, reference, output_folder=None):
#     """
#     Evaluate the performance of sound detections based on scores.
#     This function computes the AUC and PR curves for the evaluation results against the reference.

#     Args:
#         evaluation: str
#             Path to the CSV file containing evaluation results, which could be detection scores or a set of annotations.
#         reference: str 
#             Path to the CSV file containing ground-truth reference.
#         output_folder: str or None
#             The folder where to save the output CSV files. If None, the current directory is used.
#             Defaults to None.

#     Returns:
#         tuple: Two Pandas DataFrames.
#             - The first DataFrame contains detection results for each threshold and class.
#             - The second DataFrame contains classification metrics for each threshold and class, also including macro and micro averages.
#     """
    
#     # Read the detection and annotation files
#     evaluation = pd.read_csv(evaluation)
#     reference = pd.read_csv(reference)

#     # Compute AUC and PR curves
#     # auc, fpr, tpr, thresholds = compute_auc(evaluation, reference)
#     # precision, recall, thresholds_pr, ap = compute_pr_curve(evaluation, reference)
#     # Compute both AUC and PR in one pass
#     metrics = compute_auc_and_pr(evaluation, reference)

#     # Plot ROC curve
#     # if output_folder is not None:
#     #     output_folder = Path(output_folder).resolve()
#     #     output_folder.mkdir(parents=True, exist_ok=True)
#     #     plot_roc_curve(fpr, tpr, auc, output_folder / 'roc_curve.png')
#     #     plot_pr_curve(precision, recall, ap, output_folder / 'pr_curve.png')
#     if output_folder is not None:
#         output_folder = Path(output_folder).resolve()
#         output_folder.mkdir(parents=True, exist_ok=True)
#         plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], output_folder / 'roc_curve.png')
#         plot_pr_curve(metrics['precision'], metrics['recall'], metrics['average_precision'], output_folder / 'pr_curve.png')

#         print(f"ROC Curve saved to {output_folder / 'roc_curve.png'}")
#         print(f"PR Curve saved to {output_folder / 'pr_curve.png'}")
#     # print(f"AUC: {auc:.2f}")
#     # print(f"Average Precision (AP): {ap:.2f}")
#     print(f"AUC: {metrics['auc']:.2f}")
#     print(f"Average Precision (AP): {metrics['average_precision']:.2f}")

# def intervals_overlap(start1, end1, start2, end2):
#     return max(start1, start2) <= min(end1, end2)

def compare_auc_models(model_dirs, reference_file, output_folder, model_labels=None):
    plt.figure()

    if model_labels is not None and len(model_labels) != len(model_dirs):
        raise ValueError("Length of --model_labels must match --model_dirs.")
    
    for idx, model_dir in enumerate(model_dirs):
        model_dir = Path(model_dir)
        label = model_labels[idx] if model_labels else model_dir.name
        eval_file = model_dir / "detections_raw.csv"

        if not eval_file.exists():
            print(f"Skipping {label}: no detections_raw.csv found.")
            continue

        try:
            evaluation = pd.read_csv(eval_file)
            reference = pd.read_csv(reference_file)

            auc, fpr, tpr, thresholds = compute_auc(evaluation, reference)
            recision, recall, thresholds_pr, ap = compute_pr_curve(evaluation, reference)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
        except Exception as e:
            print(f"Error processing {label}: {e}")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    output_path = output_folder + "combined_roc.png"
    plt.savefig(output_path)
    print(f"ROC curve comparison saved to {output_path}")


# def compute_auc(evaluation_df, reference_df):
#     y_true = []
#     y_score = []

#     for _, pred_row in evaluation_df.iterrows():
#         overlap = find_overlap(pred_row, reference_df)
#         y_true.append(1 if overlap else 0)
#         y_score.append(pred_row['score'])  # assumes 'score' exists in evaluation_df

#     auc = roc_auc_score(y_true, y_score)
#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
#     return auc, fpr, tpr, thresholds

# def compute_pr_curve(evaluation_df, reference_df):
#     y_true = []
#     y_score = []

#     for _, pred_row in evaluation_df.iterrows():
#         overlap = find_overlap(pred_row, reference_df)
#         y_true.append(1 if overlap else 0)
#         y_score.append(pred_row['score'])  # class 1 score

#     precision, recall, thresholds = precision_recall_curve(y_true, y_score)
#     ap = average_precision_score(y_true, y_score)

#     return precision, recall, thresholds, ap

def intervals_overlap(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)

def find_overlap(row, df2, start_col='start', end_col='end'):
    start1, end1 = row[start_col], row[end_col]
    for _, row2 in df2.iterrows():
        start2, end2 = row2[start_col], row2[end_col]
        if intervals_overlap(start1, end1, start2, end2):
            return True
    return False