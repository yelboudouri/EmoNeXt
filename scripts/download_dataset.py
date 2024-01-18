import os
import shutil
import subprocess
import tarfile

import os
import pandas as pd
import numpy as np
from PIL import Image

# Set up Kaggle API credentials
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

if not os.path.isfile(kaggle_json_path):
    username = input("Kaggle username: ")
    api_key = input("Kaggle API key: ")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

    # Write the credentials to the kaggle.json file
    with open(kaggle_json_path, "w") as file:
        file.write(f'{{"username":"{username}","key":"{api_key}"}}')

    # Set file permissions to read and write for the owner only
    os.chmod(kaggle_json_path, 0o600)

# Importing kaggle will authenticate automatically
import kaggle

# Command to authenticate and download the dataset
api_command = "kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge -f fer2013.tar.gz"

# Execute the command
try:
    subprocess.run(api_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(
        "An error occurred while downloading the dataset. Please double-check your Kaggle API key."
    )
    os.remove(kaggle_json_path)

print("Preparing dataset..")
with tarfile.open("fer2013.tar.gz", "r") as tar:
    tar.extractall("fer2013")

output_folder_path = "FER2013"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("fer2013/fer2013/fer2013.csv")

# Define a dictionary to map emotion codes to labels
emotion_labels = {
    "0": "Angry",
    "1": "Disgust",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprise",
    "6": "Neutral",
}

# Create the output folders and subfolders if they do not exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
for usage in ["train", "val", "test"]:
    usage_folder_path = os.path.join(output_folder_path, usage)
    if not os.path.exists(usage_folder_path):
        os.makedirs(usage_folder_path)
    for label in emotion_labels.values():
        subfolder_path = os.path.join(usage_folder_path, label)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

# Loop over each row in the DataFrame
for index, row in df.iterrows():
    # Extract the image data from the row
    pixels = row["pixels"].split()
    img_data = [int(pixel) for pixel in pixels]
    img_array = np.array(img_data).reshape(48, 48)
    img = Image.fromarray(img_array.astype("uint8"), "L")

    # Get the emotion label and determine the output subfolder based on the Usage column
    emotion_label = emotion_labels[str(row["emotion"])]
    if row["Usage"] == "Training":
        output_subfolder_path = os.path.join(output_folder_path, "train", emotion_label)
    elif row["Usage"] == "PublicTest":
        output_subfolder_path = os.path.join(output_folder_path, "val", emotion_label)
    else:
        output_subfolder_path = os.path.join(output_folder_path, "test", emotion_label)

    # Save the image to the output subfolder
    output_file_path = os.path.join(output_subfolder_path, f"{index}.jpg")
    img.save(output_file_path)

print("Deleting temporary files..")
os.remove("fer2013.tar.gz")
shutil.rmtree("fer2013")
