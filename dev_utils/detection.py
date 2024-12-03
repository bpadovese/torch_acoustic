import pandas as pd
import torch

def filter_by_label(detecitons, labels):
    """
    Filters the input DataFrame by specified label(s).
    
    Args:
        detections: pandas DataFrame
            A DataFrame containing the results data.
        labels: list or integer 
            A list of labels to filter by.

    Returns:
        pandas.DataFrame: 
            A DataFrame containing only the detections with the specified labels.

    Example:

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'filename': ['file1.wav', 'file2.wav', 'file3.wav'],
    ...     'start': [0.0, 20.0, 40.0],
    ...     'end': [20.0, 40.0, 60.0],
    ...     'label': [0, 1, 2],
    ...     'score': [0.6, 0.8, 0.7]
    ... })
    >>> filtered_df = filter_by_label(df, 1)
    >>> filtered_df = filtered_df.reset_index(drop=True)
    >>> filtered_df.equals(pd.DataFrame({
    ...     'filename': ['file2.wav'],
    ...     'start': [20.0],
    ...     'end': [40.0],
    ...     'label': [1],
    ...     'score': [0.8]
    ... }))
    True

    """
    if not isinstance(labels, list):
        labels = [labels]
    return detecitons[detecitons['label'].isin(labels)]


def apply_detection_threshold(predictions, threshold=0.5, highest_score_only=False):
    """
    Filters out detection scores below or at a specified threshold and returns a list of tuples, where each tuple consists of a label and the score itself.
    
    Args:
        predictions: list of floats
            The list of predictions.
        threshold: float
            The threshold below which scores are filtered out. Default is 0.5.
        highest_score_only: bool
            If True, only the highest score is returned. Default is False.

    Returns:
        list of tuples: Each tuple contains a label and the score itself.

    Examples:

    >>> apply_detection_threshold([0.2, 0.7, 0.3], 0.5)
    [(1, 0.7)]
    >>> apply_detection_threshold([0.6, 0.4, 0.8], 0.55)
    [(0, 0.6), (2, 0.8)]
    >>> apply_detection_threshold([0.6, 0.4, 0.8], 0.6, True)
    [(2, 0.8)]
    """
    # Convert scores to tensor if not already
    # if not isinstance(predictions, torch.Tensor):
    #     predictions = torch.tensor(predictions)
    filtered_predictions = [(label, score.item()) for label, score in enumerate(predictions) if score >= threshold]

    filtered_predictions = [(label, score) for label, score in enumerate(predictions) if score >= threshold]
    
    if highest_score_only and filtered_predictions:
        return [max(filtered_predictions, key=lambda item: item[1])]
    else:
        return filtered_predictions


def filter_by_threshold(detections, threshold=0.5, highest_score_only=False):
    """
    Filters out detection scores below a specified threshold and returns a DataFrame with the remaining detections. 
    
    Args:
        detections: dict
            The dictionary with the detections.
        threshold: float
            The threshold below which scores are filtered out. Default is 0.5.
        highest_score_only: bool
            If True, only the highest score is returned. Default is False.

    Returns:
        pd.DataFrame: DataFrame with the remaining detections.

    Examples:
    
    >>> detections = {
    ...    'filename': ['file1', 'file2'],
    ...    'start': [0, 1],
    ...    'end': [1, 2],
    ...    'score': [[0.2, 0.7, 0.3], [0.1, 0.4, 0.8]]
    ... }
    >>> filter_by_threshold(detections, 0.5)
      filename  start  end  label  score
    0    file1      0    1      1    0.7
    1    file2      1    2      2    0.8
    """
    # create an empty list to store the filtered output
    filtered_output = {'filename': [], 'start': [], 'end': [], 'label': [], 'score': []}

    for filename, start, end, prediction in zip(detections['filename'], detections['start'], detections['end'], detections['prediction']):
        detections = apply_detection_threshold(prediction, threshold, highest_score_only)
        # for each score that passes the threshold, duplicate the filename, start, end, and add the corresponding label
        # if there is no score above the threshold, exclude the segment from the 
        for label, score in detections:
            filtered_output['filename'].append(filename)
            filtered_output['start'].append(start)
            filtered_output['end'].append(end)
            filtered_output['label'].append(label)
            filtered_output['score'].append(score)

    return pd.DataFrame(filtered_output)

def merge_overlapping_detections(detections_df):
    """ Merge overlapping or adjacent detections with the same label.

        The score of the merged detection is computed as the average of the individual detection scores.

        Note: The detections are assumed to be sorted by start time in chronological order.

        Args:
            detections_df: pandas DataFrame
                Dataframe with detections. It should have the following columns:
                - 'filename': The name of the file containing the detection.
                - 'start': The start time of the detection in seconds.
                - 'end': The end time of the detection in seconds.
                - 'label': The label associated with the detection.
                - 'score': The score associated with the detection.
        
        Returns:
            merged: pandas DataFrame
                DataFrame with the merged detections.

        Example:
            Given a DataFrame with the following format:

            +----------+-------+-----+-------+-------+
            | filename | start | end | label | score |
            +----------+-------+-----+-------+-------+
            | file1    | 0     | 5   | 0     | 1     |
            +----------+-------+-----+-------+-------+
            | file1    | 3     | 7   | 0     | 2     |
            +----------+-------+-----+-------+-------+
            | file2    | 0     | 5   | 1     | 3     |
            +----------+-------+-----+-------+-------+

            The function would return:
            
            +----------+-------+-----+-------+-------+
            | filename | start | end | label | score |
            +----------+-------+-----+-------+-------+
            | file1    | 0     | 7   | 0     | 1.5   |
            +----------+-------+-----+-------+-------+
            | file2    | 0     | 5   | 1     | 3     |
            +----------+-------+-----+-------+-------+

        >>> import pandas as pd
        >>> detections_df = pd.DataFrame([
        ...     {'filename': 'file1', 'start': 0, 'end': 5, 'label': 0, 'score': 1},
        ...     {'filename': 'file1', 'start': 3, 'end': 7, 'label': 0, 'score': 2},
        ...     {'filename': 'file2', 'start': 0, 'end': 5, 'label': 1, 'score': 3}
        ... ])
        >>> merged = merge_overlapping_detections(detections_df)
        >>> merged.to_dict('records')
        [{'filename': 'file1', 'start': 0, 'end': 7, 'label': 0, 'score': 1.5}, {'filename': 'file2', 'start': 0, 'end': 5, 'label': 1, 'score': 3.0}]
    """
    detections = detections_df.to_dict('records')

    if len(detections) <= 1:
        return detections_df
    
    merged_detections = [detections[0]]

    for i in range(1,len(detections)):
        # detections do not overlap, nor are they adjacent nor they are from the same label
        if detections[i]['start'] > merged_detections[-1]['end'] or detections[i]['filename'] != merged_detections[-1]['filename'] or detections[i]['label'] != merged_detections[-1]['label']:
            merged_detections.append(detections[i])
        # detections overlap, or adjacent to one another
        else:
            # determine if the score is a list or a single number
            if isinstance(merged_detections[-1]['score'], list):
                overlap = merged_detections[-1]['end'] - detections[i]['start'] + 1 # amount of overlap

                # handle the scores
                merged_scores = merged_detections[-1]['score'][: -overlap]  # get the non-overlapping part from the first detection
                overlap_scores_merged = merged_detections[-1]['score'][-overlap:]  # get the overlapping part from the first detection
                overlap_scores_new = detections[i]['score'][:overlap]  # get the overlapping part from the second detection
                overlap_scores_avg = [(x + y) / 2 for x, y in zip(overlap_scores_merged, overlap_scores_new)]  # average the overlapping scores
                
                merged_scores += overlap_scores_avg  # add the averaged overlapping scores to the merged scores
                merged_scores += detections[i]['score'][overlap:]  # add the non-overlapping part from the second detection to the merged scores
            else:
                # if score is a single number, just average it as before
                merged_scores = (merged_detections[-1]['score'] + detections[i]['score']) / 2
            
            # create the new merged detection
            merged_detection = {
                'filename': detections[i]['filename'], 
                'start': merged_detections[-1]['start'], 
                'end': detections[i]['end'], 
                'label': detections[i]['label'], # add the label of the merged detection
                'score': merged_scores
            }

            merged_detections[-1] = merged_detection #replace

    return pd.DataFrame(merged_detections)
