import re
from typing import Set
import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")


def _parse_list_answer(answer_str: str) -> Set[str]:
    """
    Parses a multi-line string (typically from an <answer> tag) into a set of items.
    It expects items to potentially be prefixed with "- ".
    """
    items = set()
    if not answer_str.strip():  # Handle empty or whitespace-only answers
        return items
    for line in answer_str.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            items.add(line[2:].strip())
        elif line:  # Add non-empty lines even if they don't start with "- "
            items.add(line)
    return items


def _evaluate_move_quality(fen: str, move_uci: str, best_move_uci: str) -> float:
    """
    Evaluates the quality of a chess move compared to the best move.

    Args:
        fen: FEN string representation of the board position
        move_uci: UCI string of the move to evaluate
        best_move_uci: UCI string of the best move

    Returns:
        A score between 0.0 and 1.0, where 1.0 means it's the best move
    """
    # If it's already the best move, return 1.0
    if move_uci == best_move_uci:
        return 1.0

    try:
        # Create a board from the FEN position
        board = chess.Board(fen)

        # Check if the moves are valid
        try:
            best_move = chess.Move.from_uci(best_move_uci)
            candidate_move = chess.Move.from_uci(move_uci)
        except ValueError:
            # Invalid UCI string
            return 0.0

        # Check if moves are legal
        if not board.is_legal(candidate_move) or not board.is_legal(best_move):
            return 0.0

        # Use chess.engine to evaluate the position after both moves

        # Evaluate position after best move
        board.push(best_move)
        best_result = engine.analyse(board, chess.engine.Limit(time=0.1))
        best_score = best_result["score"].white().score(mate_score=10000)
        board.pop()

        # Evaluate position after candidate move
        board.push(candidate_move)
        candidate_result = engine.analyse(board, chess.engine.Limit(time=0.1))
        candidate_score = candidate_result["score"].white().score(mate_score=10000)

        # Calculate the difference in centipawns
        score_diff = abs(best_score - candidate_score)

        # Debug info to understand actual score differences
        print(
            f"Best move score: {best_score}, Candidate score: {candidate_score}, Diff: {score_diff}"
        )

        # Scale the score: 0 difference = 1.0, 100+ centipawn difference = 0.0
        # Using 100 instead of 1000 to have more meaningful intermediate values
        scaled_score = max(0.0, 1.0 - (score_diff / 100))
        return scaled_score

    except Exception as e:
        print(f"Error evaluating move quality: {e}")
        return 0.0  # Default to 0 if we can't evaluate


def calculate_reward(
    llm_response: str, target: str, question_type: str, fen: str = None
) -> float:
    """
    Calculates a reward for an LLM's response based on its format and correctness
    compared to a target answer.

    Reward scale:
    -1.0: Incorrect overall format (<think>/<answer> structure violation).
     0.0: Correct format but the worst possible answer (e.g., completely wrong content).
     ...  Values between 0.0 and 1.0 for partially correct answers (for list types).
     1.0: Correct format and the best possible (exact match) answer.

    Args:
        llm_response: The full string response from the LLM.
        target: The ground truth string expected within the <answer> tags.
        question_type: The type of question that was asked (e.g., "list_file_pieces").
        fen: Optional FEN string of the board position (needed for move evaluation)

    Returns:
        A float reward score.
    """

    # 1. Check overall format using regex
    # Pattern: ^<think>\n.*</think>\n\n<answer>\n(.*)\n</answer>$ (DOTALL)
    # The (.*?) captures the content within the <answer> tags.
    format_pattern = re.compile(
        r"^<think>\n.*</think>\n\n<answer>(.*)</answer>$", re.DOTALL
    )
    match = format_pattern.match(llm_response)

    if not match:
        return -1.0

    llm_extracted_answer = match.group(1).strip()
    # Assuming target is already the clean, stripped content expected inside <answer> tags.
    # If target might have leading/trailing newlines from generation, strip it too.
    target = target.strip()

    # Define categories of question types based on expected output format
    # These should align with the types defined in chess_queries.py
    LIST_OUTPUT_QUERIES = {
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
        "list_piece_legal_moves",
        "list_all_white_legal_moves",
        "list_all_black_legal_moves",
    }

    EXACT_MATCH_STRING_QUERIES = {
        "piece_on_square",  # Target: "<Color> <piece_type> on <square>" or empty string
    }

    CHESS_MOVE_QUERIES = {
        "best_move",  # Target: UCI string (e.g., "e2e4")
    }

    BINARY_MOVE_QUERIES = {
        "best_of_2_moves",  # Target: UCI string (the better of the two moves)
    }

    YES_NO_QUERIES = {
        "is_legal_move",  # Target: "Yes" or "No"
    }

    if question_type in LIST_OUTPUT_QUERIES:
        llm_items = _parse_list_answer(llm_extracted_answer)
        target_items = _parse_list_answer(target)

        if not target_items and not llm_items:  # Both correctly empty
            return 1.0
        # If target is empty but LLM provided items, it's a bad answer (hallucination)
        if not target_items and llm_items:
            return 0.0
        # If LLM provided nothing but target expected items, also bad.
        # This will be handled by Jaccard index naturally evaluating to 0.

        # Calculate Jaccard Index: |Intersection(A, B)| / |Union(A, B)|
        # Union(A, B) = |A| + |B| - |Intersection(A, B)|
        intersection_size = len(llm_items.intersection(target_items))
        union_size = len(llm_items) + len(target_items) - intersection_size

        if union_size == 0:  # Should be covered by "not target_items and not llm_items"
            return 1.0 if intersection_size == 0 else 0.0  # Both empty vs. error
        else:
            return intersection_size / union_size

    elif question_type in EXACT_MATCH_STRING_QUERIES:
        # For these types, any deviation from target is considered the "worst" (0.0)
        # given the format is already confirmed correct.
        if llm_extracted_answer == target:
            return 1.0
        else:
            return 0.0

    elif question_type in BINARY_MOVE_QUERIES:
        # Special handling for chess move evaluation - binary result only
        special_case_messages = [
            "Could not determine best move",
            "No legal moves available",
        ]

        # If the target is a special message, require exact match
        if target in special_case_messages:
            return 1.0 if llm_extracted_answer == target else 0.0

        # If the LLM gave a special message but target is an actual move, score 0
        if llm_extracted_answer in special_case_messages:
            return 0.0

        # For best_of_2_moves, only return 1.0 for exact match with best move
        return 1.0 if llm_extracted_answer == target else 0.0

    elif question_type in CHESS_MOVE_QUERIES:
        # Special handling for chess move evaluation with graduated scoring
        special_case_messages = [
            "Could not determine best move",
            "No legal moves available",
        ]

        # If the target is a special message, require exact match
        if target in special_case_messages:
            return 1.0 if llm_extracted_answer == target else 0.0

        # If the LLM gave a special message but target is an actual move, score 0
        if llm_extracted_answer in special_case_messages:
            return 0.0

        return _evaluate_move_quality(fen, llm_extracted_answer, target)

    elif question_type in YES_NO_QUERIES:
        # Case-insensitive comparison for "Yes" or "No"
        if llm_extracted_answer.lower() == target.lower():
            return 1.0
        else:
            return 0.0
    else:
        # This block is for question_types that are not categorized above.
        # If the format was correct, but we don't know how to score the content.
        print(
            f"Warning: Unhandled question_type '{question_type}' in reward function. Format was correct, but content scoring is not defined. Returning 0.0."
        )
        return 0.0
