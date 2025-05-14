import chess
import chess.engine
from typing import List, Tuple, Optional, Dict, Any
import random
import uuid

# --- Helper Functions ---

# Mapping from python-chess piece type constants to lowercase names
PIECE_TYPE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _get_squares_on_file(file_char: str) -> List[chess.Square]:
    """Returns a list of square indices for a given file."""
    file_index = chess.FILE_NAMES.index(file_char.lower())
    return [chess.square(file_index, rank_index) for rank_index in range(8)]


def _get_squares_on_rank(rank_char: str) -> List[chess.Square]:
    """Returns a list of square indices for a given rank."""
    rank_index = chess.RANK_NAMES.index(rank_char)
    return [chess.square(file_index, rank_index) for file_index in range(8)]


def _get_squares_on_diagonal(
    start_sq_idx: chess.Square, end_sq_idx: chess.Square
) -> List[chess.Square]:
    """
    Generates all square indices along the diagonal from start_sq to end_sq (inclusive).
    Assumes start_sq and end_sq are validated to be on the same diagonal.
    """
    squares = []
    start_file, start_rank = chess.square_file(start_sq_idx), chess.square_rank(
        start_sq_idx
    )
    end_file, end_rank = chess.square_file(end_sq_idx), chess.square_rank(end_sq_idx)

    # Determine step direction
    file_step = 1 if end_file > start_file else -1 if end_file < start_file else 0
    rank_step = 1 if end_rank > start_rank else -1 if end_rank < start_rank else 0

    # Handle single square case (technically not a diagonal)
    if file_step == 0 and rank_step == 0:
        return [start_sq_idx]

    current_file, current_rank = start_file, start_rank
    while True:
        sq = chess.square(current_file, current_rank)
        squares.append(sq)
        if current_file == end_file and current_rank == end_rank:
            break
        current_file += file_step
        current_rank += rank_step
        # Safety break if something goes wrong (shouldn't happen with validated input)
        if not (0 <= current_file <= 7 and 0 <= current_rank <= 7):
            raise RuntimeError(
                f"Diagonal generation went out of bounds from {chess.square_name(start_sq_idx)} to {chess.square_name(end_sq_idx)}"
            )

    return squares


# Define piece type priority for sorting (lower number = higher priority)
PIECE_TYPE_PRIORITY = {
    chess.KING: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
    chess.PAWN: 5,
}


def _filter_pieces(
    board: chess.Board,
    color: Optional[bool] = None,
    piece_type: Optional[chess.PieceType] = None,
) -> List[Tuple[chess.Piece, chess.Square]]:
    """Filters pieces on the board by color and/or piece type."""
    results = []
    squares_to_check = chess.SQUARES  # Check all squares initially

    # Optimization: If both filters are provided, use board.pieces()
    if color is not None and piece_type is not None:
        squares_to_check = board.pieces(piece_type, color)
        for sq in squares_to_check:
            piece = board.piece_at(sq)
            if piece:  # Safety check
                results.append((piece, sq))
        # Apply sorting even in the optimized path: Color (W>B) > Piece Type > Square
        results.sort(
            key=lambda item: (
                not item[0].color,
                PIECE_TYPE_PRIORITY[item[0].piece_type],
                item[1],
            )
        )
        return results  # Return early after optimized path

    # General case: Iterate through all squares or piece_map
    for sq in squares_to_check:
        piece = board.piece_at(sq)
        if piece:
            matches_color = color is None or piece.color == color
            matches_type = piece_type is None or piece.piece_type == piece_type
            if matches_color and matches_type:
                results.append((piece, sq))

    # Sort results based on priority: Color (W>B) > Piece Type > Square Index
    results.sort(
        key=lambda item: (
            not item[0].color,
            PIECE_TYPE_PRIORITY[item[0].piece_type],
            item[1],
        )
    )
    return results


def _format_piece(piece: chess.Piece, square_index: chess.Square) -> str:
    """Formats a piece and its location into the desired string."""
    color_name = chess.COLOR_NAMES[piece.color].capitalize()
    piece_name = PIECE_TYPE_NAMES.get(piece.piece_type, "unknown")
    square_name = chess.square_name(square_index)
    return f"- {color_name} {piece_name} on {square_name}"


def _format_piece_no_bullet(piece: chess.Piece, square_index: chess.Square) -> str:
    """Formats a piece and its location into the desired string without bullet point."""
    color_name = chess.COLOR_NAMES[piece.color].capitalize()
    piece_name = PIECE_TYPE_NAMES.get(piece.piece_type, "unknown")
    square_name = chess.square_name(square_index)
    return f"{color_name} {piece_name} on {square_name}"


def _format_move(move: chess.Move) -> str:
    """Formats a move into the desired string."""
    return f"- {move.uci()}"


# --- Main Function ---

SUPPORTED_QUERY_TYPES = {
    # Location-Based (require param)
    "list_file_pieces",
    "list_rank_pieces",
    "list_diagonal_pieces",
    # Piece-Type Based (no param)
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
    # Single Square Queries (require param)
    "piece_on_square",
    "list_piece_legal_moves",
    # Move Queries (require param/no param)
    "is_legal_move",  # requires param (e.g., "e2e4")
    "list_all_white_legal_moves",  # no param
    "list_all_black_legal_moves",  # no param
    "best_of_2_moves",  # requires param (e.g., "e2e4-d7d5")
    "best_move",  # no param
}

LOCATION_QUERIES = {"list_file_pieces", "list_rank_pieces", "list_diagonal_pieces"}
PIECE_TYPE_QUERIES = {
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
}
SINGLE_SQUARE_QUERIES = {"piece_on_square", "list_piece_legal_moves"}
MOVE_VALIDATION_QUERIES = {"is_legal_move"}
ALL_LEGAL_MOVES_QUERIES = {"list_all_white_legal_moves", "list_all_black_legal_moves"}
BEST_MOVE_QUERIES = {"best_move"}
BEST_OF_2_QUERIES = {"best_of_2_moves"}

PARAM_REQUIRED_QUERIES = (
    LOCATION_QUERIES
    | SINGLE_SQUARE_QUERIES
    | MOVE_VALIDATION_QUERIES
    | BEST_OF_2_QUERIES
)
PARAM_NOT_ALLOWED_QUERIES = (
    PIECE_TYPE_QUERIES | ALL_LEGAL_MOVES_QUERIES | BEST_MOVE_QUERIES
)


# Map query types to piece types (None means all types)
QUERY_TO_PIECE_TYPE = {
    "pawns": chess.PAWN,
    "knights": chess.KNIGHT,
    "bishops": chess.BISHOP,
    "rooks": chess.ROOK,
    "queens": chess.QUEEN,
    "kings": chess.KING,
    "pieces": None,  # Represents queries like list_white_pieces, list_all_pieces
}


def get_target(
    board: chess.Board, question_type: str, question_param: Optional[str] = None
) -> str:
    """
    Analyzes a chess board position based on a structural query and returns a
    formatted multi-line string representing the result.

    Args:
        board: A chess.Board object representing the position.
        question_type: A string identifying the type of query (e.g.,
                       'list_file_pieces', 'list_white_pawns', 'is_legal_move').
        question_param: An optional parameter required for certain queries
                        (e.g., file letter, rank number, square name, move UCI).
                        If not provided for queries that need parameters, a random
                        appropriate value will be generated.

    Returns:
        A formatted multi-line string representing the query result.
        The format depends on the query type:
        - Piece lists: "- <Color> <piece_type> on <square>" per piece.
        - Move lists: "- <move_uci>" per move.
        - piece_on_square: "- <Color> <piece_type> on <square>" or empty string.
        - is_legal_move: "- True" or "- False".
        Returns an empty string if the query yields no results (e.g., no pieces found,
        no legal moves, empty square specified).

    Raises:
        ValueError: If the question_type is unsupported, if required parameters
                    are missing/malformed, or if unexpected parameters are provided.
    """
    if question_type not in SUPPORTED_QUERY_TYPES:
        raise ValueError(
            f"Unsupported question_type: '{question_type}'. Supported types are: {', '.join(sorted(SUPPORTED_QUERY_TYPES))}"
        )

    # 1. Input Validation and Parameter Handling
    if question_type in PARAM_REQUIRED_QUERIES:
        if not question_param:
            # Generate a random valid parameter based on the query type
            if question_type == "list_file_pieces":
                question_param = random.choice(chess.FILE_NAMES)
            elif question_type == "list_rank_pieces":
                question_param = random.choice(chess.RANK_NAMES)
            elif question_type == "list_diagonal_pieces":
                # Generate a random diagonal (need two squares on same diagonal)
                file_diff = random.randint(
                    2, 7
                )  # Ensure a real diagonal (not same square)
                rank_diff = file_diff  # Same diff for a diagonal
                start_file = random.randint(0, 7 - file_diff)  # Ensure end is on board
                start_rank = random.randint(0, 7 - rank_diff)
                end_file = start_file + file_diff
                end_rank = start_rank + rank_diff
                start_sq = chess.square_name(chess.square(start_file, start_rank))
                end_sq = chess.square_name(chess.square(end_file, end_rank))
                question_param = f"{start_sq}-{end_sq}"
            elif question_type == "piece_on_square":
                # Randomly select between occupied and empty squares (50% chance each)
                occupied_squares = [
                    chess.square_name(sq)
                    for sq in chess.SQUARES
                    if board.piece_at(sq) is not None
                ]
                empty_squares = [
                    chess.square_name(sq)
                    for sq in chess.SQUARES
                    if board.piece_at(sq) is None
                ]

                if random.random() < 0.5 and occupied_squares:
                    # 50% chance of picking an occupied square
                    question_param = random.choice(occupied_squares)
                elif empty_squares:
                    # 50% chance of picking an empty square
                    question_param = random.choice(empty_squares)
                else:
                    # If all squares are occupied or empty, pick any square
                    question_param = chess.square_name(random.choice(chess.SQUARES))
            elif question_type == "list_piece_legal_moves":
                # Find a piece that has legal moves
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and any(
                        move.from_square == sq for move in board.legal_moves
                    ):
                        question_param = chess.square_name(sq)
                        break
                else:
                    # If no pieces with legal moves, use any random piece
                    occupied_squares = [
                        sq for sq in chess.SQUARES if board.piece_at(sq) is not None
                    ]
                    if occupied_squares:
                        question_param = chess.square_name(
                            random.choice(occupied_squares)
                        )
                    else:
                        # If no pieces (unlikely), use any random square
                        question_param = chess.square_name(random.choice(chess.SQUARES))
            elif question_type == "is_legal_move":
                # Randomly select between legal and illegal moves (50% chance each)
                legal_moves = list(board.legal_moves)

                if random.random() < 0.5 and legal_moves:
                    # 50% chance of picking a legal move
                    question_param = random.choice(legal_moves).uci()
                else:
                    # 50% chance of picking a likely illegal move
                    # Generate random move that's likely illegal
                    while True:
                        from_sq = chess.square_name(random.choice(chess.SQUARES))
                        to_sq = chess.square_name(random.choice(chess.SQUARES))
                        move_uci = f"{from_sq}{to_sq}"

                        # Check if it's a legal move
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if move not in board.legal_moves:
                                question_param = move_uci
                                break
                        except ValueError:
                            # Invalid UCI format, try again
                            continue

                        # If we've tried 10 times and still haven't found an illegal move,
                        # just use the current one
                        if "question_param" not in locals():
                            question_param = move_uci
                            break
            elif question_type == "best_of_2_moves":
                # Pick two different legal moves if possible
                legal_moves = list(board.legal_moves)
                if len(legal_moves) >= 2:
                    try:
                        # Attempt to get best and worst moves using Stockfish
                        best_move, worst_move = _find_best_and_worst_moves(board)
                        if best_move and worst_move:
                            # Randomly decide order of presentation
                            if random.choice([True, False]):
                                # Best move first
                                question_param = f"{best_move.uci()}-{worst_move.uci()}"
                            else:
                                # Worst move first
                                question_param = f"{worst_move.uci()}-{best_move.uci()}"
                        else:
                            # If Stockfish analysis failed, just pick random moves
                            move1, move2 = random.sample(legal_moves, 2)
                            question_param = f"{move1.uci()}-{move2.uci()}"
                    except Exception:
                        # Fall back to random selection if best/worst determination fails
                        move1, move2 = random.sample(legal_moves, 2)
                        question_param = f"{move1.uci()}-{move2.uci()}"
                elif len(legal_moves) == 1:
                    # If only one legal move, use it and a random invalid one
                    move1 = legal_moves[0].uci()
                    from_sq = chess.square_name(random.choice(chess.SQUARES))
                    to_sq = chess.square_name(random.choice(chess.SQUARES))
                    move2 = f"{from_sq}{to_sq}"
                    if move2 == move1:  # Ensure they're different
                        to_sq = chess.square_name((chess.parse_square(to_sq) + 1) % 64)
                        move2 = f"{from_sq}{to_sq}"
                    # Randomly decide order
                    if random.choice([True, False]):
                        question_param = f"{move1}-{move2}"
                    else:
                        question_param = f"{move2}-{move1}"
                else:
                    # If no legal moves, generate two random moves
                    from_sq1 = chess.square_name(random.choice(chess.SQUARES))
                    to_sq1 = chess.square_name(random.choice(chess.SQUARES))
                    move1 = f"{from_sq1}{to_sq1}"
                    from_sq2 = chess.square_name(random.choice(chess.SQUARES))
                    to_sq2 = chess.square_name(random.choice(chess.SQUARES))
                    move2 = f"{from_sq2}{to_sq2}"
                    question_param = f"{move1}-{move2}"
    elif question_type in PARAM_NOT_ALLOWED_QUERIES:
        if question_param is not None:
            raise ValueError(
                f"Query type '{question_type}' does not accept a question_param, but received '{question_param}'."
            )
    # else: Some future query might need optional params? Currently none.

    pieces_to_format: List[Tuple[chess.Piece, chess.Square]] = []
    target_squares: Optional[List[chess.Square]] = None
    target_square_idx: Optional[chess.Square] = None
    target_move: Optional[chess.Move] = None
    moves_to_format: List[chess.Move] = []

    # --- Location-Based Queries ---
    if question_type in LOCATION_QUERIES:
        if question_type == "list_file_pieces":
            param_lower = question_param.lower()
            if len(param_lower) != 1 or param_lower not in chess.FILE_NAMES:
                raise ValueError(
                    f"Invalid file parameter: '{question_param}'. Must be a single letter 'a' through 'h'."
                )
            target_squares = _get_squares_on_file(param_lower)
        elif question_type == "list_rank_pieces":
            if len(question_param) != 1 or question_param not in chess.RANK_NAMES:
                raise ValueError(
                    f"Invalid rank parameter: '{question_param}'. Must be a single digit '1' through '8'."
                )
            target_squares = _get_squares_on_rank(question_param)
        elif question_type == "list_diagonal_pieces":
            if "-" not in question_param:
                raise ValueError(
                    f"Invalid diagonal parameter format: '{question_param}'. Expected format 'square1-square2' (e.g., 'a1-h8')."
                )
            parts = question_param.split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid diagonal parameter format: '{question_param}'. Must contain exactly one hyphen."
                )
            start_sq_name, end_sq_name = (
                parts[0].strip().lower(),
                parts[1].strip().lower(),
            )

            try:
                start_sq_idx = chess.parse_square(start_sq_name)
                end_sq_idx = chess.parse_square(end_sq_name)
            except ValueError as e:
                raise ValueError(
                    f"Invalid square name in diagonal parameter '{question_param}': {e}"
                ) from e

            start_file, start_rank = chess.square_file(start_sq_idx), chess.square_rank(
                start_sq_idx
            )
            end_file, end_rank = chess.square_file(end_sq_idx), chess.square_rank(
                end_sq_idx
            )

            if start_sq_idx == end_sq_idx:
                raise ValueError(
                    f"Diagonal parameter '{question_param}' specifies the same start and end square."
                )

            if abs(start_file - end_file) != abs(start_rank - end_rank):
                raise ValueError(
                    f"Squares '{start_sq_name}' and '{end_sq_name}' are not on the same diagonal."
                )

            target_squares = _get_squares_on_diagonal(start_sq_idx, end_sq_idx)

        # Gather pieces for location queries
        if target_squares is not None:
            for sq in target_squares:
                piece = board.piece_at(sq)
                if piece:
                    pieces_to_format.append((piece, sq))
            # Sort results by square index for consistency within the location
            pieces_to_format.sort(key=lambda item: item[1])

    # --- Piece-Type Based Queries ---
    elif question_type in PIECE_TYPE_QUERIES:
        color_filter: Optional[bool] = None
        if "white" in question_type:
            color_filter = chess.WHITE
        elif "black" in question_type:
            color_filter = chess.BLACK
        # 'all' implies color_filter remains None

        piece_type_filter: Optional[chess.PieceType] = None
        type_key = question_type.split("_")[-1]  # e.g., "pawns", "knights", "pieces"
        if type_key in QUERY_TO_PIECE_TYPE:
            piece_type_filter = QUERY_TO_PIECE_TYPE[type_key]
        else:
            # This case should not happen if SUPPORTED_QUERY_TYPES is correct
            raise ValueError(
                f"Could not determine piece type for query '{question_type}'"
            )

        pieces_to_format = _filter_pieces(board, color_filter, piece_type_filter)
        # _filter_pieces already sorts correctly

    # --- Single Square Queries ---
    elif question_type in SINGLE_SQUARE_QUERIES:
        try:
            target_square_idx = chess.parse_square(question_param.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid square parameter: '{question_param}'. {e}"
            ) from e

        piece = board.piece_at(target_square_idx)

        if question_type == "piece_on_square":
            if piece:
                # Use non-bullet point formatting for single piece queries
                return _format_piece_no_bullet(piece, target_square_idx)
            return ""  # Empty string for empty square
        elif question_type == "list_piece_legal_moves":
            if piece:  # Only list moves if a piece is present
                for move in board.legal_moves:
                    if move.from_square == target_square_idx:
                        moves_to_format.append(move)
                # Sort moves alphabetically by UCI string
                moves_to_format.sort(key=lambda m: m.uci())
            # else: moves_to_format remains empty

    # --- Move Validation Queries ---
    elif question_type in MOVE_VALIDATION_QUERIES:
        if question_type == "is_legal_move":
            try:
                target_move = chess.Move.from_uci(question_param.lower())
            except ValueError as e:
                raise ValueError(
                    f"Invalid move parameter format: '{question_param}'. Expected UCI format (e.g., 'e2e4'). {e}"
                ) from e

            # Result is formatted directly later

    # --- All Legal Moves Queries ---
    elif question_type in ALL_LEGAL_MOVES_QUERIES:
        target_color = (
            chess.WHITE
            if question_type == "list_all_white_legal_moves"
            else chess.BLACK
        )
        for move in board.legal_moves:
            # Check the color of the piece being moved
            piece = board.piece_at(move.from_square)
            if piece and piece.color == target_color:
                moves_to_format.append(move)
        # Sort moves alphabetically by UCI string
        moves_to_format.sort(key=lambda m: m.uci())

    # --- Best of 2 Moves Query ---
    elif question_type in BEST_OF_2_QUERIES:
        if "-" not in question_param:
            raise ValueError(
                f"Invalid best_of_2_moves parameter format: '{question_param}'. Expected format 'move1-move2' (e.g., 'e2e4-d7d5')."
            )
        parts = question_param.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid best_of_2_moves parameter format: '{question_param}'. Must contain exactly one hyphen."
            )
        move1_uci, move2_uci = parts[0].strip().lower(), parts[1].strip().lower()

        try:
            move1 = chess.Move.from_uci(move1_uci)
            move2 = chess.Move.from_uci(move2_uci)
        except ValueError as e:
            raise ValueError(
                f"Invalid move UCI in best_of_2_moves parameter '{question_param}': {e}"
            ) from e

        legal_moves = list(board.legal_moves)
        if move1 not in legal_moves:
            raise ValueError(
                f"Move '{move1.uci()}' is not legal in the current position."
            )
        if move2 not in legal_moves:
            raise ValueError(
                f"Move '{move2.uci()}' is not legal in the current position."
            )

        # Determine which is the best move using Stockfish
        stockfish_path = "stockfish"  # Adjust if necessary
        try:
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                # Evaluate both moves
                board_copy1 = board.copy()
                board_copy1.push(move1)
                result1 = engine.analyse(board_copy1, chess.engine.Limit(time=0.05))
                score1 = result1["score"].relative.score(mate_score=10000)

                board_copy2 = board.copy()
                board_copy2.push(move2)
                result2 = engine.analyse(board_copy2, chess.engine.Limit(time=0.05))
                score2 = result2["score"].relative.score(mate_score=10000)

                # Compare scores to determine better move
                if score1 >= score2:
                    target_move = move1
                else:
                    target_move = move2
        except Exception as e:
            # If Stockfish fails, return a message
            return "Could not determine best move"

    # --- Best Move Query (Stockfish) ---
    elif question_type in BEST_MOVE_QUERIES:
        # NOTE: Requires Stockfish engine executable in PATH
        stockfish_path = "stockfish"  # Adjust if necessary
        try:
            # Use a context manager to ensure the engine quits properly
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                # Analyze the position with a short time limit (e.g., 0.1 seconds)
                # Adjust time limit as needed
                result = engine.play(board, chess.engine.Limit(time=0.1))
                if result.move:
                    target_move = result.move
                else:
                    return "No legal moves available"
        except Exception as e:
            # If Stockfish fails, return a message
            return "Could not determine best move"

    # 3. Format Results
    formatted_lines = []
    if pieces_to_format:
        formatted_lines = [_format_piece(piece, sq) for piece, sq in pieces_to_format]
    elif moves_to_format:
        formatted_lines = [_format_move(move) for move in moves_to_format]
    elif (
        question_type == "is_legal_move" and target_move is not None
    ):  # Special case formatting
        is_legal = target_move in board.legal_moves
        formatted_lines = [f"{'Yes' if is_legal else 'No'}"]
    elif (
        question_type in BEST_OF_2_QUERIES or question_type in BEST_MOVE_QUERIES
    ) and target_move:
        # Format single best move UCI
        formatted_lines = [target_move.uci()]

    # 4. Return Joined String
    return "\n".join(formatted_lines)


def _find_best_and_worst_moves(
    board: chess.Board,
) -> Tuple[Optional[chess.Move], Optional[chess.Move]]:
    """
    Uses Stockfish to evaluate moves and find the best and worst legal moves.

    Args:
        board: A chess.Board object representing the position.

    Returns:
        A tuple of (best_move, worst_move) or (None, None) if evaluation fails.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, None

    if len(legal_moves) == 1:
        # If only one legal move, it's both best and worst
        move = legal_moves[0]
        return move, move

    # Try using Stockfish for evaluation
    stockfish_path = "stockfish"  # Adjust if necessary
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Evaluate all legal moves
            move_evaluations = []
            for move in legal_moves:
                board_copy = board.copy()
                board_copy.push(move)
                result = engine.analyse(board_copy, chess.engine.Limit(time=0.05))
                score = result["score"].relative.score(mate_score=10000)
                move_evaluations.append((move, score))

            # Sort by evaluation score
            move_evaluations.sort(key=lambda x: x[1], reverse=True)

            best_move = move_evaluations[0][0]
            worst_move = move_evaluations[-1][0]

            # Add attributes to identify the moves
            setattr(best_move, "is_best", True)
            setattr(worst_move, "is_worst", True)

            return best_move, worst_move
    except Exception as e:
        # In case Stockfish fails or isn't available, return None
        return None, None


def get_prompt(
    board: chess.Board, question_type: str, question_param: Optional[str] = None
) -> str:
    """
    Generates a prompt for a given chess position and query type.

    Args:
        board: A chess.Board object representing the position.
        question_type: A string identifying the type of query.
        question_param: An optional parameter required for certain queries.
                        If not provided for queries that need parameters, a random
                        appropriate value will be generated.

    Returns:
        A formatted string representing the complete prompt to use for LLM generation.

    Raises:
        ValueError: If the question_type is unsupported, if required parameters
                   are missing/malformed, or if unexpected parameters are provided.
    """
    if question_type not in SUPPORTED_QUERY_TYPES:
        raise ValueError(
            f"Unsupported question_type: '{question_type}'. Supported types are: {', '.join(sorted(SUPPORTED_QUERY_TYPES))}"
        )

    # Initialize answer_format differently based on question type to avoid default value issues
    if question_type == "list_all_white_legal_moves":
        answer_format = """<answer>
- a2a3
- a2a4
- b1a3
- b1c3
- b2b3
- b2b4
- c2c3
- c2c4
- d2d3
- d2d4
- e2e3
- e2e4
- f2f3
- f2f4
- g1f3
- g1h3
- g2g3
- g2g4
- h2h3
- h2h4
</answer>

If there are no legal moves for White, provide empty answer tags:

<answer>
</answer>"""
    elif question_type == "list_all_black_legal_moves":
        answer_format = """<answer>
- a7a5
- a7a6
- b7b5
- b7b6
- b8a6
- b8c6
- c7c5
- c7c6
- d7d5
- d7d6
- e7e5
- e7e6
- f7f5
- f7f6
- g7g5
- g7g6
- g8f6
- g8h6
- h7h5
- h7h6
</answer>

If there are no legal moves for Black, provide empty answer tags:

<answer>
</answer>"""
    else:
        answer_format = (
            "<answer>\nDefault format for unexpected question type\n</answer>"
        )

    # Validate parameters
    if question_type in PARAM_REQUIRED_QUERIES:
        if not question_param:
            # Generate a random valid parameter based on the query type
            if question_type == "list_file_pieces":
                question_param = random.choice(chess.FILE_NAMES)
            elif question_type == "list_rank_pieces":
                question_param = random.choice(chess.RANK_NAMES)
            elif question_type == "list_diagonal_pieces":
                # Generate a random diagonal (need two squares on same diagonal)
                file_diff = random.randint(
                    2, 7
                )  # Ensure a real diagonal (not same square)
                rank_diff = file_diff  # Same diff for a diagonal
                start_file = random.randint(0, 7 - file_diff)  # Ensure end is on board
                start_rank = random.randint(0, 7 - rank_diff)
                end_file = start_file + file_diff
                end_rank = start_rank + rank_diff
                start_sq = chess.square_name(chess.square(start_file, start_rank))
                end_sq = chess.square_name(chess.square(end_file, end_rank))
                question_param = f"{start_sq}-{end_sq}"
            elif question_type == "piece_on_square":
                # Randomly select between occupied and empty squares (50% chance each)
                occupied_squares = [
                    chess.square_name(sq)
                    for sq in chess.SQUARES
                    if board.piece_at(sq) is not None
                ]
                empty_squares = [
                    chess.square_name(sq)
                    for sq in chess.SQUARES
                    if board.piece_at(sq) is None
                ]

                if random.random() < 0.5 and occupied_squares:
                    # 50% chance of picking an occupied square
                    question_param = random.choice(occupied_squares)
                elif empty_squares:
                    # 50% chance of picking an empty square
                    question_param = random.choice(empty_squares)
                else:
                    # If all squares are occupied or empty, pick any square
                    question_param = chess.square_name(random.choice(chess.SQUARES))
            elif question_type == "list_piece_legal_moves":
                # Find a piece that has legal moves
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and any(
                        move.from_square == sq for move in board.legal_moves
                    ):
                        question_param = chess.square_name(sq)
                        break
                else:
                    # If no pieces with legal moves, use any random piece
                    occupied_squares = [
                        sq for sq in chess.SQUARES if board.piece_at(sq) is not None
                    ]
                    if occupied_squares:
                        question_param = chess.square_name(
                            random.choice(occupied_squares)
                        )
                    else:
                        # If no pieces (unlikely), use any random square
                        question_param = chess.square_name(random.choice(chess.SQUARES))
            elif question_type == "is_legal_move":
                # Randomly select between legal and illegal moves (50% chance each)
                legal_moves = list(board.legal_moves)

                if random.random() < 0.5 and legal_moves:
                    # 50% chance of picking a legal move
                    question_param = random.choice(legal_moves).uci()
                else:
                    # 50% chance of picking a likely illegal move
                    # Generate random move that's likely illegal
                    while True:
                        from_sq = chess.square_name(random.choice(chess.SQUARES))
                        to_sq = chess.square_name(random.choice(chess.SQUARES))
                        move_uci = f"{from_sq}{to_sq}"

                        # Check if it's a legal move
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if move not in board.legal_moves:
                                question_param = move_uci
                                break
                        except ValueError:
                            # Invalid UCI format, try again
                            continue

                        # If we've tried 10 times and still haven't found an illegal move,
                        # just use the current one
                        if "question_param" not in locals():
                            question_param = move_uci
                            break
            elif question_type == "best_of_2_moves":
                # Pick two different legal moves if possible
                legal_moves = list(board.legal_moves)
                if len(legal_moves) >= 2:
                    try:
                        # Attempt to get best and worst moves using Stockfish
                        best_move, worst_move = _find_best_and_worst_moves(board)
                        if best_move and worst_move:
                            # Randomly decide order of presentation
                            if random.choice([True, False]):
                                # Best move first
                                question_param = f"{best_move.uci()}-{worst_move.uci()}"
                            else:
                                # Worst move first
                                question_param = f"{worst_move.uci()}-{best_move.uci()}"
                        else:
                            # If Stockfish analysis failed, just pick random moves
                            move1, move2 = random.sample(legal_moves, 2)
                            question_param = f"{move1.uci()}-{move2.uci()}"
                    except Exception:
                        # Fall back to random selection if best/worst determination fails
                        move1, move2 = random.sample(legal_moves, 2)
                        question_param = f"{move1.uci()}-{move2.uci()}"
                elif len(legal_moves) == 1:
                    # If only one legal move, use it and a random invalid one
                    move1 = legal_moves[0].uci()
                    from_sq = chess.square_name(random.choice(chess.SQUARES))
                    to_sq = chess.square_name(random.choice(chess.SQUARES))
                    move2 = f"{from_sq}{to_sq}"
                    if move2 == move1:  # Ensure they're different
                        to_sq = chess.square_name((chess.parse_square(to_sq) + 1) % 64)
                        move2 = f"{from_sq}{to_sq}"
                    # Randomly decide order
                    if random.choice([True, False]):
                        question_param = f"{move1}-{move2}"
                    else:
                        question_param = f"{move2}-{move1}"
                else:
                    # If no legal moves, generate two random moves
                    from_sq1 = chess.square_name(random.choice(chess.SQUARES))
                    to_sq1 = chess.square_name(random.choice(chess.SQUARES))
                    move1 = f"{from_sq1}{to_sq1}"
                    from_sq2 = chess.square_name(random.choice(chess.SQUARES))
                    to_sq2 = chess.square_name(random.choice(chess.SQUARES))
                    move2 = f"{from_sq2}{to_sq2}"
                    question_param = f"{move1}-{move2}"
    elif question_type in PARAM_NOT_ALLOWED_QUERIES:
        if question_param is not None:
            raise ValueError(
                f"Query type '{question_type}' does not accept a question_param, but received '{question_param}'."
            )

    # Base template
    base_template = """You are the greatest chess grandmaster in the world.

{QUESTION}

Answer in <answer> tags.

<fen>
{FEN}
</fen>"""

    # <rules>
    # - Please provide your answer in the following example format:

    # {ANSWER_FORMAT}

    # - Please state the exact problem you need to solve and the FEN in the very first reasoning step.

    # - You must always think before you answer.

    # - You must think like a human.
    # </rules>

    # Get FEN string from board
    fen = board.fen()

    # Initialize question - will be set in the if/elif blocks
    question = f"Please analyze the position and answer the query: {question_type}"

    # Define question templates and answer formats for each query type
    if question_type == "list_file_pieces":
        file_letter = question_param.upper()
        question_options = [
            f"Please list all pieces located on the {file_letter} file.",
            f"Which pieces are on the {file_letter} file?",
            f"Show all the pieces on the {file_letter} file.",
            f"List the positions of all pieces on the {file_letter} file.",
            f"What pieces are currently on the {file_letter} file?",
            f"Display all pieces occupying the {file_letter} file.",
            f"Provide a list of pieces on the {file_letter} file.",
            f"Identify all pieces placed on the {file_letter} file.",
            f"Output the pieces that are on the {file_letter} file.",
            f"Which squares on the {file_letter} file are occupied, and by which pieces?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
- White rook on a1
- White pawn on a2
- Black pawn on a7
- Black rook on a8
</answer>

If no pieces are on the file, provide empty answer tags:

<answer>
</answer>"""

    elif question_type == "list_rank_pieces":
        rank_number = question_param
        question_options = [
            f"Please list all pieces located on the {rank_number} rank.",
            f"Which pieces are on the {rank_number} rank?",
            f"Show all the pieces on the {rank_number} rank.",
            f"List the positions of all pieces on the {rank_number} rank.",
            f"What pieces are currently on the {rank_number} rank?",
            f"Display all pieces occupying the {rank_number} rank.",
            f"Provide a list of pieces on the {rank_number} rank.",
            f"Identify all pieces placed on the {rank_number} rank.",
            f"Output the pieces that are on the {rank_number} rank.",
            f"Which squares on the {rank_number} rank are occupied, and by which pieces?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
- White rook on a1
- White knight on b1
- White bishop on c1
- White queen on d1
- White king on e1
- White bishop on f1
- White knight on g1
- White rook on h1
</answer>

If no pieces are on the rank, provide empty answer tags:

<answer>
</answer>"""

    elif question_type == "list_diagonal_pieces":
        diagonal = question_param.replace("-", "–").upper()  # Using en dash
        question_options = [
            f"Please list all pieces located on the {diagonal} diagonal.",
            f"Which pieces are on the {diagonal} diagonal?",
            f"Show all the pieces on the {diagonal} diagonal.",
            f"List the positions of all pieces on the {diagonal} diagonal.",
            f"What pieces are currently on the {diagonal} diagonal?",
            f"Display all pieces occupying the {diagonal} diagonal.",
            f"Provide a list of pieces on the {diagonal} diagonal.",
            f"Identify all pieces placed on the {diagonal} diagonal.",
            f"Output the pieces that are on the {diagonal} diagonal.",
            f"Which squares on the {diagonal} diagonal are occupied, and by which pieces?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
- White rook on a1
- White pawn on b2
- Black pawn on g7
- Black rook on h8
</answer>

If no pieces are on the diagonal, provide empty answer tags:

<answer>
</answer>"""

    elif question_type == "is_legal_move":
        move = question_param.lower()
        question_options = [
            f"Is {move} a legal move?",
            f"Can you tell me if {move} is legal?",
            f"Is {move} allowed in this position?",
            f"Would {move} be a valid move here?",
            f"Am I allowed to play {move}?",
            f"Is it possible to play {move}?",
            f"Can you legally make the move {move}?",
            f"Is {move} permitted by the rules?",
            f"Could I play {move} right now?",
            f"Would {move} be considered a legal move?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
Yes
</answer>

OR

<answer>
No
</answer>"""

    elif question_type == "piece_on_square":
        square = question_param.lower()
        question_options = [
            f"What piece is on {square}?",
            # f"What's sitting on {square}?",
            # f"Can you identify the piece on {square}?",
            # f"What's located on {square}?",
            # f"What's the piece at {square}?",
            # f"Is there a piece on {square}, and if so, what is it?",
            # f"Tell me what piece is at {square}.",
            # f"Who's occupying {square}?",
            # f"What's on {square} right now?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
White knight on e4
</answer>

If there is no piece on the square, provide empty answer tags:

<answer>
No piece on b4
</answer>"""

    elif question_type == "list_piece_legal_moves":
        square = question_param.lower()
        # Get the piece at this square for better question format
        piece = board.piece_at(chess.parse_square(square))
        piece_desc = ""
        if piece:
            color_name = chess.COLOR_NAMES[piece.color].capitalize()
            piece_name = PIECE_TYPE_NAMES.get(piece.piece_type, "")
            piece_desc = f"{color_name}'s {piece_name}"
        else:
            piece_desc = "piece"  # Fallback if no piece

        question_options = [
            f"List all the legal moves {piece_desc} on {square} can make.",
            f"What are all the legal moves for the {piece_desc} on {square}?",
            f"Where can the {piece_desc} on {square} go?",
            f"Show me every legal move the {piece_desc} on {square} can make.",
            f"Can you list all valid moves for the {piece_desc} on {square}?",
            f"What options does the {piece_desc} on {square} have?",
            f"Which squares can the {piece_desc} on {square} legally move to?",
            f"Tell me all possible destinations for the {piece_desc} on {square}.",
            f"What are the movement options for the {piece_desc} at {square}?",
            f"Can the {piece_desc} on {square} move anywhere? If so, where?",
            f"What moves can the {piece_desc} on {square} legally make from here?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
- e2e3
- e2e4
</answer>

If there are no legal moves or no piece on the square, provide empty answer tags:

<answer>
</answer>"""

    elif question_type.startswith("list_white_"):
        piece_type = question_type.split("_")[-1]
        if piece_type == "pieces":
            piece_str = "pieces"
        else:
            piece_str = piece_type  # Keep original (e.g., "bishops", "kings")

        question_options = [
            f"Please list all of White's {piece_str} on the board.",
            f"Which of White's {piece_str} are currently on the board?",
            f"Show all of White's {piece_str} on the chessboard.",
            f"List the positions of all White {piece_str} on the board.",
            f"What White {piece_str} are currently present on the board?",
            f"Display all squares occupied by White's {piece_str}.",
            f"Provide a complete list of White's {piece_str} on the board.",
            f"Identify all of White's {piece_str} currently placed on the board.",
            f"Output the full list of White's {piece_str} and their positions.",
            f"Which squares are occupied by White's {piece_str}?",
        ]
        question = random.choice(question_options)

        # Set the appropriate answer format based on piece type
        if piece_type == "kings":
            answer_format = """<answer>
- White king on e1
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "queens":
            answer_format = """<answer>
- White queen on d1
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "rooks":
            answer_format = """<answer>
- White rook on a1
- White rook on h1
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "bishops":
            answer_format = """<answer>
- White bishop on c1
- White bishop on f1
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "knights":
            answer_format = """<answer>
- White knight on b1
- White knight on g1
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "pawns":
            answer_format = """<answer>
- White pawn on a2
- White pawn on b2
- White pawn on c2
- White pawn on d2
- White pawn on e2
- White pawn on f2
- White pawn on g2
- White pawn on h2
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "pieces":
            answer_format = """<answer>
- White king on e1
- White queen on d1
- White rook on a1
- White rook on h1
- White bishop on c1
- White bishop on f1
- White knight on b1
- White knight on g1
- White pawn on a2
- White pawn on b2
- White pawn on c2
- White pawn on d2
- White pawn on e2
- White pawn on f2
- White pawn on g2
- White pawn on h2
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""

    elif question_type.startswith("list_black_"):
        piece_type = question_type.split("_")[-1]
        if piece_type == "pieces":
            piece_str = "pieces"
        else:
            piece_str = piece_type  # Keep original (e.g., "bishops", "kings")

        question_options = [
            f"Please list all of Black's {piece_str} on the board.",
            f"Which of Black's {piece_str} are currently on the board?",
            f"Show all of Black's {piece_str} on the chessboard.",
            f"List the positions of all Black {piece_str} on the board.",
            f"What Black {piece_str} are currently present on the board?",
            f"Display all squares occupied by Black's {piece_str}.",
            f"Provide a complete list of Black's {piece_str} on the board.",
            f"Identify all of Black's {piece_str} currently placed on the board.",
            f"Output the full list of Black's {piece_str} and their positions.",
            f"Which squares are occupied by Black's {piece_str}?",
        ]
        question = random.choice(question_options)

        # Set the appropriate answer format based on piece type
        if piece_type == "kings":
            answer_format = """<answer>
- Black king on e8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "queens":
            answer_format = """<answer>
- Black queen on d8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "rooks":
            answer_format = """<answer>
- Black rook on a8
- Black rook on h8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "bishops":
            answer_format = """<answer>
- Black bishop on c8
- Black bishop on f8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "knights":
            answer_format = """<answer>
- Black knight on b8
- Black knight on g8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "pawns":
            answer_format = """<answer>
- Black pawn on a7
- Black pawn on b7
- Black pawn on c7
- Black pawn on d7
- Black pawn on e7
- Black pawn on f7
- Black pawn on g7
- Black pawn on h7
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "pieces":
            answer_format = """<answer>
- Black king on e8
- Black queen on d8
- Black rook on a8
- Black rook on h8
- Black bishop on c8
- Black bishop on f8
- Black knight on b8
- Black knight on g8
- Black pawn on a7
- Black pawn on b7
- Black pawn on c7
- Black pawn on d7
- Black pawn on e7
- Black pawn on f7
- Black pawn on g7
- Black pawn on h7
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""

    elif question_type == "list_all_white_legal_moves":
        question_options = [
            "What are all of White's legal moves?",
            "Show me every move White is allowed to make.",
            "List all possible moves for White in this position.",
            "What moves are available to White?",
            "What can White legally play here?",
            "Which moves can White make from this position?",
            "Tell me all the legal options White has.",
            "What are White's valid moves at the moment?",
            "What are all the legal moves for White on this turn?",
        ]
        question = random.choice(question_options)
        # answer_format is already defined at the beginning of the function

    elif question_type == "list_all_black_legal_moves":
        question_options = [
            "What are all of Black's legal moves?",
            "Show me every move Black is allowed to make.",
            "List all possible moves for Black in this position.",
            "What moves are available to Black?",
            "What can Black legally play here?",
            "Which moves can Black make from this position?",
            "Tell me all the legal options Black has.",
            "What are Black's valid moves at the moment?",
            "What are all the legal moves for Black on this turn?",
        ]
        question = random.choice(question_options)
        # answer_format is already defined at the beginning of the function

    elif question_type.startswith("list_all_"):
        piece_type = question_type.split("_")[-1]
        if piece_type == "pieces":
            piece_str = "pieces"
        else:
            piece_str = piece_type  # Keep original (e.g., "bishops", "kings")

        question_options = [
            f"Please list all of the {piece_str} on the board.",
            f"Which {piece_str} are currently on the board?",
            f"Show all the {piece_str} on the chessboard.",
            f"List the positions of all {piece_str} on the board.",
            f"What {piece_str} are currently present on the board?",
            f"Display all {piece_str} occupying squares on the board.",
            f"Provide a complete list of {piece_str} on the board.",
            f"Identify all {piece_str} currently placed on the board.",
            f"Output the full list of {piece_str} and their positions.",
            f"Which squares are occupied by {piece_str}?",
        ]
        question = random.choice(question_options)

        # Set the appropriate answer format based on piece type
        if piece_type == "kings":
            answer_format = """<answer>
- White king on e1
- Black king on e8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "queens":
            answer_format = """<answer>
- White queen on d1
- Black queen on d8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "rooks":
            answer_format = """<answer>
- White rook on a1
- White rook on h1
- Black rook on a8
- Black rook on h8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "bishops":
            answer_format = """<answer>
- White bishop on c1
- White bishop on f1
- Black bishop on c8
- Black bishop on f8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "knights":
            answer_format = """<answer>
- White knight on b1
- White knight on g1
- Black knight on b8
- Black knight on g8
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "pawns":
            answer_format = """<answer>
- White pawn on a2
- White pawn on b2
- White pawn on c2
- White pawn on d2
- White pawn on e2
- White pawn on f2
- White pawn on g2
- White pawn on h2
- Black pawn on a7
- Black pawn on b7
- Black pawn on c7
- Black pawn on d7
- Black pawn on e7
- Black pawn on f7
- Black pawn on g7
- Black pawn on h7
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""
        elif piece_type == "pieces":
            answer_format = """<answer>
- White king on e1
- White queen on d1
- White rook on a1
- White rook on h1
- White bishop on c1
- White bishop on f1
- White knight on b1
- White knight on g1
- White pawn on a2
- White pawn on b2
- White pawn on c2
- White pawn on d2
- White pawn on e2
- White pawn on f2
- White pawn on g2
- White pawn on h2
- Black king on e8
- Black queen on d8
- Black rook on a8
- Black rook on h8
- Black bishop on c8
- Black bishop on f8
- Black knight on b8
- Black knight on g8
- Black pawn on a7
- Black pawn on b7
- Black pawn on c7
- Black pawn on d7
- Black pawn on e7
- Black pawn on f7
- Black pawn on g7
- Black pawn on h7
</answer>

If no such pieces exist, provide empty answer tags:

<answer>
</answer>"""

    elif question_type == "best_of_2_moves":
        moves = question_param.lower().replace("-", " or ")
        question_options = [
            f"Which move is better: {moves}?",
            f"Between {moves}, which is stronger?",
            f"Which is the better option: {moves}?",
            f"Which move leads to a better position: {moves}?",
            f"Which of these moves is more advantageous: {moves}?",
            f"Which would you play: {moves}?",
            f"Of the two moves {moves}, which one is better?",
            f"If you had to choose between {moves}, which would be best?",
            f"Compare {moves} and tell me which is superior.",
            f"Evaluate the two moves {moves} and tell me which is best.",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
e2e4
</answer>"""

    elif question_type == "best_move":
        # Determine the side to move
        color = "White" if board.turn == chess.WHITE else "Black"
        question_options = [
            f"What is the next best move for {color}?",
            f"What's the strongest move for {color} right now?",
            f"What is {color}'s best move in this position?",
            f"What is the optimal move for {color}?",
            f"What's the most effective move for {color} to make now?",
        ]
        question = random.choice(question_options)
        answer_format = """<answer>
e2e4
</answer>"""
    else:
        # Add a catch-all for any question types that might have been missed
        question = (
            f"Please analyze the position according to the {question_type} query."
        )
        answer_format = """<answer>
Response for unhandled question type
</answer>"""
        print(
            f"Warning: Question type '{question_type}' does not have specific formatting in get_prompt()."
        )

    # Replace template variables
    prompt = base_template.format(
        QUESTION=question,
        FEN=fen,
        ANSWER_FORMAT=answer_format,
    ).rstrip()

    return prompt


def get_sample(
    board: chess.Board, question_type: str, question_param: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a sample for a given chess position and query type, containing
    the board PGN, target answer, question type, and formatted prompt.

    Args:
        board: A chess.Board object representing the position.
        question_type: A string identifying the type of query.
        question_param: An optional parameter required for certain queries.
                        If not provided for queries that need parameters, a random
                        appropriate value will be generated.

    Returns:
        A dictionary containing:
        - 'pgn': The PGN representation of the board
        - 'target': The target answer (from get_target)
        - 'question_type': The original question type
        - 'prompt': The formatted prompt (from get_prompt)

    Raises:
        ValueError: If the question_type is unsupported, if required parameters
                   are missing/malformed, or if unexpected parameters are provided.
    """
    if question_type not in SUPPORTED_QUERY_TYPES:
        raise ValueError(
            f"Unsupported question_type: '{question_type}'. Supported types are: {', '.join(sorted(SUPPORTED_QUERY_TYPES))}"
        )

    # Generate a parameter if needed and not provided
    if question_type in PARAM_REQUIRED_QUERIES and not question_param:
        # This will be the same parameter generation logic used in both functions
        if question_type == "list_file_pieces":
            question_param = random.choice(chess.FILE_NAMES)
        elif question_type == "list_rank_pieces":
            question_param = random.choice(chess.RANK_NAMES)
        elif question_type == "list_diagonal_pieces":
            file_diff = random.randint(2, 7)
            rank_diff = file_diff
            start_file = random.randint(0, 7 - file_diff)
            start_rank = random.randint(0, 7 - rank_diff)
            end_file = start_file + file_diff
            end_rank = start_rank + rank_diff
            start_sq = chess.square_name(chess.square(start_file, start_rank))
            end_sq = chess.square_name(chess.square(end_file, end_rank))
            question_param = f"{start_sq}-{end_sq}"
        elif question_type == "piece_on_square":
            # Randomly select between occupied and empty squares (50% chance each)
            occupied_squares = [
                chess.square_name(sq)
                for sq in chess.SQUARES
                if board.piece_at(sq) is not None
            ]
            empty_squares = [
                chess.square_name(sq)
                for sq in chess.SQUARES
                if board.piece_at(sq) is None
            ]

            if random.random() < 0.5 and occupied_squares:
                # 50% chance of picking an occupied square
                question_param = random.choice(occupied_squares)
            elif empty_squares:
                # 50% chance of picking an empty square
                question_param = random.choice(empty_squares)
            else:
                # If all squares are occupied or empty, pick any square
                question_param = chess.square_name(random.choice(chess.SQUARES))
        elif question_type == "list_piece_legal_moves":
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and any(move.from_square == sq for move in board.legal_moves):
                    question_param = chess.square_name(sq)
                    break
            else:
                occupied_squares = [
                    sq for sq in chess.SQUARES if board.piece_at(sq) is not None
                ]
                if occupied_squares:
                    question_param = chess.square_name(random.choice(occupied_squares))
                else:
                    question_param = chess.square_name(random.choice(chess.SQUARES))
        elif question_type == "is_legal_move":
            # Randomly select between legal and illegal moves (50% chance each)
            legal_moves = list(board.legal_moves)

            if random.random() < 0.5 and legal_moves:
                # 50% chance of picking a legal move
                question_param = random.choice(legal_moves).uci()
            else:
                # 50% chance of picking a likely illegal move
                # Generate random move that's likely illegal
                while True:
                    from_sq = chess.square_name(random.choice(chess.SQUARES))
                    to_sq = chess.square_name(random.choice(chess.SQUARES))
                    move_uci = f"{from_sq}{to_sq}"

                    # Check if it's a legal move
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move not in board.legal_moves:
                            question_param = move_uci
                            break
                    except ValueError:
                        # Invalid UCI format, try again
                        continue

                    # If we've tried 10 times and still haven't found an illegal move,
                    # just use the current one
                    if "question_param" not in locals():
                        question_param = move_uci
                        break
        elif question_type == "best_of_2_moves":
            # Pick two different legal moves if possible
            legal_moves = list(board.legal_moves)
            if len(legal_moves) >= 2:
                try:
                    # Attempt to get best and worst moves using Stockfish
                    best_move, worst_move = _find_best_and_worst_moves(board)
                    if best_move and worst_move:
                        # Randomly decide order of presentation
                        if random.choice([True, False]):
                            # Best move first
                            question_param = f"{best_move.uci()}-{worst_move.uci()}"
                        else:
                            # Worst move first
                            question_param = f"{worst_move.uci()}-{best_move.uci()}"
                    else:
                        # If Stockfish analysis failed, just pick random moves
                        move1, move2 = random.sample(legal_moves, 2)
                        question_param = f"{move1.uci()}-{move2.uci()}"
                except Exception:
                    # Fall back to random selection if best/worst determination fails
                    move1, move2 = random.sample(legal_moves, 2)
                    question_param = f"{move1.uci()}-{move2.uci()}"
            elif len(legal_moves) == 1:
                # If only one legal move, use it and a random invalid one
                move1 = legal_moves[0].uci()
                from_sq = chess.square_name(random.choice(chess.SQUARES))
                to_sq = chess.square_name(random.choice(chess.SQUARES))
                move2 = f"{from_sq}{to_sq}"
                if move2 == move1:  # Ensure they're different
                    to_sq = chess.square_name((chess.parse_square(to_sq) + 1) % 64)
                    move2 = f"{from_sq}{to_sq}"
                # Randomly decide order
                if random.choice([True, False]):
                    question_param = f"{move1}-{move2}"
                else:
                    question_param = f"{move2}-{move1}"
            else:
                # If no legal moves, generate two random moves
                from_sq1 = chess.square_name(random.choice(chess.SQUARES))
                to_sq1 = chess.square_name(random.choice(chess.SQUARES))
                move1 = f"{from_sq1}{to_sq1}"
                from_sq2 = chess.square_name(random.choice(chess.SQUARES))
                to_sq2 = chess.square_name(random.choice(chess.SQUARES))
                move2 = f"{from_sq2}{to_sq2}"
                question_param = f"{move1}-{move2}"

    # Get target answer and prompt with the same parameter
    target = get_target(board, question_type, question_param)
    prompt = get_prompt(board, question_type, question_param)

    # Generate PGN from the board
    fen = board.fen()

    # Return the sample dictionary
    return {
        "fen": fen,
        "target": target,
        "question_type": question_type,
        "prompt": prompt,
        "id": str(uuid.uuid4()),
    }
