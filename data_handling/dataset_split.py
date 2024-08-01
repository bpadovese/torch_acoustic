import pandas as pd

def load_lk_csv():
    # Load the CSV file
    df = pd.read_csv('annot_LimeKiln-Encounters_man_det.csv')  # Replace with your CSV file path
    # Filter rows where 'sound_id_species' is exactly 'KW'
    df = df[df['sound_id_species'] == 'KW']

    # Drop unnecessary columns
    df = df[['path', 'start', 'end', 'sound_id_species']]
    # Rename the columns
    df = df.rename(columns={'path': 'filename', 'sound_id_species': 'label'})
    # Reset index
    df = df.reset_index(drop=True)
    return df

def load_filtered_dataset():
    df = pd.read_csv('annotations/LK_annotations.csv')
    return df

# Function to check if a row belongs to a test folder
def is_test_row(filename, test_folders):
    for folder in test_folders:
        if folder in filename:
            return True
    return False

test_folders = ["20200913", "20190921", "20190827"]
df_filtered = load_filtered_dataset()

# Split the dataset into training and testing sets
train_df = df_filtered[~df_filtered['filename'].apply(lambda x: is_test_row(x, test_folders))]
test_df = df_filtered[df_filtered['filename'].apply(lambda x: is_test_row(x, test_folders))]

# You can now proceed with saving these to CSVs or using them directly for your model training and evaluation
train_df.to_csv('training_set_KW.csv', index=False)
test_df.to_csv('testing_set_KW.csv', index=False)

print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

# Save the filenames to separate text files
train_filenames = train_df['filename'].unique()
test_filenames = test_df['filename'].unique()

with open('training_set_KW_files.txt', 'w') as f:
    for filename in train_filenames:
        f.write(f"{filename}\n")

with open('testing_set_KW_files.txt', 'w') as f:
    for filename in test_filenames:
        f.write(f"{filename}\n")

# Save the filtered DataFrame to a new CSV file
# df_filtered.to_csv('LK_annotations.csv', index=False)