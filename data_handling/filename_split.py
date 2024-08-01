import pandas as pd
from deepechoes.utils.hdf5_helper import file_duration_table

data_dir = "Data/LK"

files = file_duration_table(data_dir, num=None)

# Function to check if a row belongs to a test folder
def is_test_row(filename, test_folders):
    for folder in test_folders:
        if folder in filename:
            return True
    return False

test_folders = ["20200913", "20190921", "20190827"]

# Split the dataset into training and testing sets
train_df = files[~files['filename'].apply(lambda x: is_test_row(x, test_folders))]
test_df = files[files['filename'].apply(lambda x: is_test_row(x, test_folders))]

# Save the filenames to separate text files
train_filenames = train_df['filename']
test_filenames = test_df['filename']

with open('training_set_KW_files.txt', 'w') as f:
    for filename in train_filenames:
        f.write(f"{filename}\n")

with open('testing_set_KW_files.txt', 'w') as f:
    for filename in test_filenames:
        f.write(f"{filename}\n")