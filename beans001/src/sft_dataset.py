# ============================================================ #

EVAL_OUTPUT_DIRECTORY = "../eval/output/local_train/beans001s"
EVAL_INPUT_DATASET_FILE = "datasets/beans001_train.json"
OUTPUT_FILE_FOR_SFT = "datasets/beans001_sft.json"

# ============================================================ #

import json
import re
from collections import Counter
from pathlib import Path

output_dir = Path(EVAL_OUTPUT_DIRECTORY)
dataset_file = Path(EVAL_INPUT_DATASET_FILE)
output_file = Path(OUTPUT_FILE_FOR_SFT)

# Load the dataset
with open(dataset_file, "r") as f:
    dataset = json.load(f)

# Create a mapping from ID to question_type
id_to_question_type = {item["id"]: item["question_type"] for item in dataset}

# Find all files that start with 1.00
sft_files = []
for file in output_dir.glob("1.00:*.json"):
    sft_files.append(file)

print(f"Found {len(sft_files)} files starting with 1.00")

# Extract IDs and count question types
combined_dataset = []
question_type_counts = Counter()

for file_path in sft_files:
    # Extract ID from filename (format: 1.00:52d5299e-f0cc-4ede-a3bf-21cb3a36a20b.json)
    match = re.search(r"1\.00:(.*?)\.json$", str(file_path))
    if not match:
        print(f"Could not extract ID from {file_path}")
        continue

    file_id = match.group(1)

    # Read the file
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # The file's data should have its own ID field, but we'll use the one from the filename
        # to match with our dataset since that's likely what was intended
        if file_id in id_to_question_type:
            # Add the question_type to the data
            question_type = id_to_question_type[file_id]
            data["question_type"] = question_type
            question_type_counts[question_type] += 1
            combined_dataset.append(data)
        else:
            print(f"ID {file_id} not found in the dataset")
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Print question type distribution
print("Question type distribution:")
for question_type, count in question_type_counts.items():
    print(f"  {question_type}: {count}")

# Save the combined dataset
with open(output_file, "w") as f:
    json.dump(combined_dataset, f, indent=2)

print(f"Saved {len(combined_dataset)} items to {output_file}")
