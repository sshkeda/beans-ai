import random

# ============================================================ #


QUERY_SAMPLE_SPACE = [
    [
        "list_file_pieces",
        "list_rank_pieces",
        "list_diagonal_pieces",
        "list_white_pieces",
        "list_white_pawns",
        "list_white_knights",
        "list_white_bishops",
        "list_white_rooks",
        "list_white_queens",
        "list_white_kings",
        "list_black_pieces",
        "list_black_pawns",
        "list_black_knights",
        "list_black_bishops",
        "list_black_rooks",
        "list_black_queens",
        "list_black_kings",
        "list_all_pieces",
        "list_all_pawns",
        "list_all_knights",
        "list_all_bishops",
        "list_all_rooks",
        "list_all_queens",
        "list_all_kings",
        "piece_on_square",
    ],
    [
        "list_piece_legal_moves",
        "is_legal_move",
        "list_all_white_legal_moves",
        "list_all_black_legal_moves",
    ],
    [
        "best_of_2_moves",
        "best_move",
    ],
]

MODEL = "beans004"
TYPE = "train"

DATASET_OUTPUT_FILE = f"datasets/{MODEL}_{TYPE}" + ".json"

COUNT = 100_000 if TYPE == "train" else 100
QUESTIONS_TO_GENERATE = ["piece_on_square"] * COUNT

print(len(QUESTIONS_TO_GENERATE))
# ============================================================ #


"""
Script to generate a dataset of chess query samples for evaluation.
"""

import json
from utils.generate_positions import generate_positions
from utils.chess_queries import get_sample, SUPPORTED_QUERY_TYPES


def create_dataset(output_file, query_types):
    """
    Create a dataset of chess queries.

    Args:
        output_file: Path to save the dataset.
        query_types: List of query types to generate samples for.
    """
    num_samples = len(query_types)
    positions = generate_positions(num_samples)
    samples = []

    # Generate samples for each position and specified query type
    count = 0
    position_index = 0

    for query_type in query_types:
        # Validate query type
        if query_type not in SUPPORTED_QUERY_TYPES:
            print(f"Warning: {query_type} is not a supported query type. Skipping.")
            continue

        # Select a position
        board = positions[position_index]
        position_index += 1

        # Try to generate a sample, retry with new board if it fails
        success = False
        retry_count = 0
        max_retries = 10

        while not success and retry_count < max_retries:
            try:
                sample = get_sample(board, query_type)
                samples.append(sample)
                count += 1
                print(f"Generated sample {count}/{num_samples}")
                success = True
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(
                        f"Failed to generate sample for {query_type} after {max_retries} attempts: {e}"
                    )
                else:
                    print(
                        f"Error generating sample for {query_type}: {e}. Retrying with new board..."
                    )
                    # Generate a new board for retry
                    board = generate_positions(1)[0]

    # Save the dataset
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    import os

    # Create the datasets directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    create_dataset(DATASET_OUTPUT_FILE, QUESTIONS_TO_GENERATE)
