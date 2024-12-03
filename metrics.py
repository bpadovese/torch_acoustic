import pandas as pd
import numpy as np
from pathlib import Path


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

def get_continuous_results(evaluation, reference, threshold=0.5):
    """
    Evaluate the True Positives (TP), False Positives (FP), and False Negatives (FN) for each unique class label
    in the ground truth DataFrame based on the evaluation DataFrame.

    This function is suitable for long, continuous segments of data and is designed to compare two sets of 
    time-stamped annotations, which could either be predictions and ground truths, or two different sets of annotations.
    
    Args:
        evaluation: pd.DataFrame
            DataFrame containing the evaluation results, which could be detection scores or another set of annotations.
            The DataFrame should have columns like 'label', 'start', 'end', etc.
        reference: pd.DataFrame
            DataFrame containing the ground truth labels, with similar structure to `predicted`.
        threshold: float
            The threshold for classifying a detection as positive or negative.
    
    Returns: dict
        A dictionary where keys are the unique class labels, and the values are dictionaries containing the counts of TP, FP, FN.
    """
    # Filter detections based on the threshold
    # predictions_threshold = evaluation[evaluation['score'] >= threshold]
    predictions_threshold = evaluation
    print(predictions_threshold)
    print(reference)
    unique_classes = reference['label'].unique().tolist()
    evaluation_results = {}
   
    for cls in unique_classes:
        tp = fp = fn = 0
        pred_filtered = predictions_threshold[predictions_threshold['label'] == cls]
        gt_filtered = reference[reference['label'] == cls]
        # We are now going to calculate the TP and FN. Note that this calculation covers instances where we have one prediction that overlaps with
        # multiple ground truths. In this case, the TP will be incremented for each ground truth
        # We are also considering Multiple preditcions for one ground truth, in this case, only one of the predictions will increment the TP.
        # Another assumption made here is that even if only 1% of the prediction falls within the ground truth it will count as a TP

        # Loop through each ground truth entry
        for _, gt_row in gt_filtered.iterrows():
            # If an overlap is found, increment TP, otherwise increment FN
            if find_overlap(gt_row, pred_filtered):
                tp += 1
            else:
                fn += 1
        
        # Calculate FP:
        # Loop through each predicted filtered entry
        for _, predicted_row in pred_filtered.iterrows():
            # If no overlap is found, increment FP
            if not find_overlap(predicted_row, gt_filtered):
                fp += 1

        evaluation_results[cls] = {
            'TP': tp,
            'FP': fp,
            'FN': fn
        }

    return evaluation_results

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

def evaluate(evaluation, reference, threshold_min=0, threshold_max=1, threshold_inc=0.05, total_time_units=None, output_folder=None):
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

    # Initialize dictionaries to store results and metrics for different thresholds
    all_threshold_results = {}
    all_threshold_metrics = {}

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
        thresholds = list(np.arange(threshold_min, threshold_max, threshold_inc))
        if thresholds[-1] != threshold_max: # Manually including the last threshold
            thresholds.append(threshold_max)
    
    for threshold in thresholds:
        threshold = round(threshold, 5) # rounding the threshold to avoid floating-point issues
        
        # Get results
        result = get_continuous_results(evaluation, reference, threshold=threshold)
        metrics_for_this_threshold = {}
    
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0
        
        # Loop through each unique class label to calculate metrics
        for label, counts in result.items():
            TP = counts['TP']
            FP = counts['FP']
            FN = counts['FN']
            
            total_TP += TP
            total_FP += FP
            total_FN += FN

            TN = None
            total_TN = None
            
            metrics = calculate_metrics(TP, FP, FN, TN, total_time_units=total_time_units)
            metrics_for_this_threshold[label] = metrics
                
        # Calculate and store macro-average metrics
        macro_avg_metrics = {metric: np.mean([x[metric] for x in metrics_for_this_threshold.values()])
                            for metric in metrics_to_calculate}
        
        metrics_for_this_threshold['macro_avg'] = macro_avg_metrics  # Store macro averages as a 'label' in the result dictionary
        
        # Calculate and store micro-average metrics
        micro_avg_metrics = calculate_metrics(total_TP, total_FP, total_FN, total_TN, total_time_units=total_time_units)
        metrics_for_this_threshold['micro_avg'] = micro_avg_metrics  # Store micro averages as a 'label' in the result dictionary

        all_threshold_metrics[threshold] = metrics_for_this_threshold
        # Store the results and metrics for this threshold
        all_threshold_results[threshold] = result

    # Convert these nested dictionaries to a Pandas DataFrame
    all_results_df = pd.DataFrame.from_dict({(i, j): all_threshold_results[i][j] 
                                             for i in all_threshold_results.keys() 
                                             for j in all_threshold_results[i].keys()}, 
                                             orient='index')

    all_metrics_df = pd.DataFrame.from_dict({(i, j): all_threshold_metrics[i][j] 
                                             for i in all_threshold_metrics.keys() 
                                             for j in all_threshold_metrics[i].keys()}, 
                                             orient='index')
    
    # Name the multi-level index columns
    all_results_df.index.names = ['threshold', 'class']
    all_metrics_df.index.names = ['threshold', 'class']

    # Save these DataFrames to CSV
    all_results_df.to_csv((output_folder / 'results.csv'), index=True)
    all_metrics_df.to_csv((output_folder / 'metrics.csv'), index=True)

    print("All results and metrics saved to {}".format(output_folder))
    
    return all_results_df, all_metrics_df


def main():
    import argparse
        
    parser = argparse.ArgumentParser(description="Evaluate the performance of sound detection.")
    parser.add_argument('evaluation', type=str, help='Path to the .csv file containing evaluation results, which may include detection scores or annotations.')
    parser.add_argument('reference', type=str, help='Path to the .csv file containing ground truth reference.')
    parser.add_argument('--threshold_min', default=0, type=float, help='Minimum threshold for detection.')
    parser.add_argument('--threshold_max', default=1, type=float, help='Maximum threshold for detection.')
    parser.add_argument('--threshold_inc', default=0.05, type=float, help='Threshold increment for each step.')
    parser.add_argument('--total_time_units', default=None, type=float, help='The total duration in arbitrary time units over which the detections were made (e.g., hours, minutes).' \
            ' This is necessary to calculate the False Positive Rate per unit time. Defaults to None.')
    parser.add_argument('--output_folder', default=None, type=str, help='Location to output the performance results. For instance: metrics/')

    
    args = parser.parse_args()
    evaluate(**vars(args))

if __name__ == "__main__":
    main()
