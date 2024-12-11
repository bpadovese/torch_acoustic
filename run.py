import torch
import numpy as np
import soundfile as sf
import json
import librosa
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import sigmoid
from torchvision import transforms
from lightning.fabric import Fabric
from pathlib import Path
from data_handling.dataset import NormalizeToRange, ConditionalResize
from torch.utils.data import DataLoader
from data_handling.spec_preprocessing import classifier_representation
from dev_utils.nn import resnet18_for_single_channel, resnet50_for_single_channel
from dev_utils.detection import filter_by_threshold, merge_overlapping_detections, filter_by_label

def output_function(batch_detections, output_folder, threshold=0.5, merge_detections=False, buffer=None, running_avg=None, labels=None, highest_score_only=False):
    batch_detections = filter_by_threshold(batch_detections, threshold, highest_score_only)
    
    if merge_detections:
        batch_detections = merge_overlapping_detections(batch_detections)
    
    return batch_detections

def segment_generator():
    
    pass 

def process_audio(file_path, config, model, batch_size=32):
    image_size = 128
    # Initialize dictionary to store lists for each attribute
    predictions_data = {
        "filename": [],
        "start": [],
        "end": [],
        "prediction": [], 
    }

    # Open the audio file with soundfile
    with sf.SoundFile(file_path) as audio_file:
        sr = audio_file.samplerate
        segment_length = int(config['duration'] * sr)  # Segment duration in samples
        total_segments = int(np.ceil(audio_file.frames / segment_length))  # Calculate total segments

        with tqdm(total=total_segments, desc=f"Processing Segments for {file_path}", leave=False) as segment_pbar:
            for i in range(total_segments):
                start_sample = i * segment_length
                end_sample = min(start_sample + segment_length, audio_file.frames)  # Ensure bounds are not exceeded
                start_time = start_sample / sr  # Convert start sample to time in seconds
                end_time = end_sample / sr      # Convert end sample to time in seconds
    
                # Read only the current segment
                audio_file.seek(start_sample)  # Move to the start of the segment
                segment = audio_file.read(end_sample - start_sample)

                # Pad if the last segment is shorter than the required duration
                if len(segment) < segment_length:
                        segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant', constant_values=0)

                # Resample the audio segment if new sample rate is provided and different from the original
                if config['sr'] is not None and config['sr'] != sr:
                    segment = librosa.resample(segment, orig_sr=sr, target_sr=config['sr'])

                # Convert the audio segment to the model's expected representation
                representation_data = classifier_representation(
                    segment, config["window"], config["step"], config['sr'], config["num_filters"], 
                    fmin=config["fmin"], fmax=config["fmax"]
                )

                transform_pipeline = transforms.Compose([
                    ConditionalResize((image_size, image_size)),
                    transforms.ToTensor(),
                ])

                # Prepare the data for model input
                input_tensor = torch.from_numpy(representation_data).unsqueeze(0).unsqueeze(0).float()  # Add batch dimension
                input_tensor = transform_pipeline(input_tensor)  # Apply all transformations

                logits = model(input_tensor)
                # cpu will move a tensor from the GPU to the CPU
                probabilities = sigmoid(logits).detach().cpu().numpy()
                # print(probabilities.squeeze())
                # Append to predictions list
                predictions_data["filename"].append(file_path.relative_to(config['audio_data']))
                predictions_data["start"].append(start_time)
                predictions_data["end"].append(end_time)
                predictions_data["prediction"].append(probabilities.squeeze()) 
            
                # Update the inner progress bar
                segment_pbar.update(1)

    return predictions_data

def load_file_list(file_list_path):
    """
    Load file paths from a .txt or .csv file into a list using Pandas.

    Parameters:
    - file_list_path: Path to the file list (either .txt or .csv).

    Returns:
    - List of file paths as strings.
    """
    file_list_path = Path(file_list_path)

    # Use Pandas to read the file
    if file_list_path.suffix in {".txt", ".csv"}:
        file_list = pd.read_csv(file_list_path, header=None).iloc[:, 0].tolist()
    else:
        raise ValueError("Unsupported file list format. Use .txt or .csv.")

    return file_list

def main(model_file, audio_data, file_list=None, output_folder=None, overwrite=True, step_size=None, threshold=0.5, labels=None, merge_detections=False, buffer=0.0):
    # Load your pre-trained model
    model = resnet18_for_single_channel()
    fabric = Fabric()
    state = fabric.load(model_file)
    # "trained_models/time_shift/sample_normalize/rb/my_model_v0.pt"

    model.load_state_dict(state["model"])
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)
    model.eval()
    

    audio_representation = "common/config_files/spec_config.json"

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.mkdir(parents=True, exist_ok=True)

    mode= 'w'
    if not overwrite:
        mode = "a"

    # Open and read the audio configuration file (e.g., JSON file with audio settings)
    with open(audio_representation, 'r') as f:
        config = json.load(f)

    # Example usage
    audio_path = Path(audio_data)
    config['audio_data'] = audio_path
    audio_files = []
    for ext in ['*.wav', '*.flac']:
        audio_files.extend(audio_path.rglob(ext))
        
    if file_list is not None:
        file_list = load_file_list(file_list)
        file_list = [audio_path / Path(file) for file in file_list]

        # Normalize and resolve all paths in the file list
        file_list_paths = {Path(f).resolve() for f in file_list}

        # Filter audio files that match the resolved paths
        audio_files = [audio_file for audio_file in audio_files if Path(audio_file).resolve() in file_list_paths]

    # Initialize tqdm with the total segment count
    with tqdm(total=len(audio_files), desc="Processing Audio Files") as pbar:
        for file_path in audio_files:
            file_predictions = process_audio(file_path, config, model)
            
            # Update tqdm for each processed file
            pbar.update(1)
            # Apply filter by threshold and print the result
            file_predictions = filter_by_threshold(file_predictions, threshold=threshold)
            
            if labels is not None:
                file_predictions = filter_by_label(file_predictions, labels=labels)

            if merge_detections:
                file_predictions = merge_overlapping_detections(file_predictions)

            header = True if mode == "w" else False
            output = (output_folder / "detections.csv")
            if not file_predictions.empty:
                # Write detections with header only for the first write
                file_predictions.to_csv(output, mode=mode, index=False, header=header)
            elif mode == 'w' or not output.exists():
                # If batch_detections is empty and mode is 'w', write an empty DataFrame to the file.
                empty_df = pd.DataFrame(columns=['filename', 'start', 'end', 'label', 'score'])
                empty_df.to_csv(output, mode='w', index=False, header=header)
            
            mode = 'a'

# Example usage
if __name__ == "__main__":
    import argparse
    import ast
    def boolean_string(s):
            if s not in {'False', 'True'}:
                raise ValueError('Not a valid boolean string')
            return s == 'True'

    def tryeval(val):
        # Literal eval does cast type safely. However, doesnt work for str, therefore the try except.
        try:
            val = ast.literal_eval(val)
        except ValueError:
            pass
        return val
        
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = tryeval(value)

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help='Path to the torch model file (*.pt)')
    parser.add_argument('audio_data', type=str, help='Path to either a folder with audio files.')
    parser.add_argument('--file_list', default=None, type=str, help='A .csv or .txt file where each row (or line) is the name of a file to detect within the audio folder. \
                        By default, all files will be processed.')
    parser.add_argument('--output_folder', default=None, type=str, help='Location to output the detections. For instance: detections/')
    parser.add_argument('--overwrite', default=True, type=boolean_string, help='Overwrites the detections, otherwise appends to it.')
    parser.add_argument('--step_size', default=None, type=float, help='Step size in seconds. If not specified, the step size is set equal to the duration of the audio representation.')
    parser.add_argument('--threshold', default=0.5, type=float, help="The threshold value used to determine the cut-off point for detections. This is a floating-point value between 0 and 1. A detection is considered positive if its score is above this threshold. The default value is 0.5.")
    parser.add_argument('--labels', type=tryeval, default=None, help="List or integer of labels to filter by. Example usage: --labels 1 or --labels [1,2,3]. Defaults to None.")
    parser.add_argument('--merge_detections', default=False, type=boolean_string, 
                    help="A flag indicating whether to merge overlapping detections into a single detection. If set to True, overlapping detections are merged. The default value is False, meaning detections are kept separate.")
    parser.add_argument('--buffer', default=0.0, type=float, 
                    help="The buffer duration to be added to each detection in seconds. This helps to extend the start and end times of each detection to include some context around the detected event. The default value is 0.0, which means no buffer is added.")
    
    args = parser.parse_args()
    main(args.model_file, args.audio_data, file_list=args.file_list, output_folder=args.output_folder, overwrite=args.overwrite, 
         step_size=args.step_size, threshold=args.threshold, labels=args.labels, merge_detections=args.merge_detections, buffer=args.buffer)