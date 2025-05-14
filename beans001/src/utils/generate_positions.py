import chess
import chess.engine
import random
from tqdm import tqdm
import time # Added for potential debugging/timing
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os # Added for cpu_count

# Helper function for canonical FEN key (position + turn, no clocks/move counts)
def canon_key(b: chess.Board) -> str:
    """Generates a canonical FEN key ignoring clocks and move numbers."""
    return b.board_fen() + (" w" if b.turn == chess.WHITE else " b")

def simulate_game(args):
    """
    Simulates a single chess game based on the provided scenario and returns
    a randomly selected FEN string from its history, or None on failure.
    Manages its own engine instance.
    """
    seed, scenario, stockfish_path = args
    rnd = random.Random(seed) # Use process-specific random generator
    white_mode, black_mode = scenario
    
    engine = None # Initialize engine to None
    try:
        # Start Stockfish engine for this process
        # Consider adding UCI options here if needed (e.g., UCI_LimitStrength)
        # options = {"Skill Level": skill} # Kept for compatibility, consider UCI_Elo
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        board = chess.Board()
        game_fens = [] # Store FEN strings

        # Reduced depths and adjusted max_plies
        if white_mode == "engine_strong" and black_mode == "engine_strong":
            max_plies = 80 # Reduced for faster strong vs strong
        elif white_mode == "random" and black_mode == "random":
            max_plies = 120 # Allow longer random games
        else:
            max_plies = 100 # Mixed/weak scenarios

        # Revised skill/depth maps
        skill_map = {"engine_strong": 20, "engine_mid": 10, "engine_weak": 1} # Use lower skill for weak
        depth_map = {"engine_strong": 8, "engine_mid": 4, "engine_weak": 2} # Significantly reduced depths
        
        moves_played = 0
        
        while not board.is_game_over(claim_draw=True) and moves_played < max_plies:
            turn_mode = white_mode if board.turn == chess.WHITE else black_mode
            move = None 
            
            # Higher chance for weak engine to play random
            if turn_mode == "engine_weak" and rnd.random() < 0.6: 
                 turn_mode = "random" 

            if turn_mode == "random":
                legal_moves = list(board.legal_moves)
                if not legal_moves: break
                move = rnd.choice(legal_moves)
            else:
                skill = skill_map.get(turn_mode, 20)
                depth = depth_map.get(turn_mode, 2) # Default to low depth
                limit = chess.engine.Limit(depth=depth) 
                options = {"Skill Level": skill} # Pass skill level option
                
                # Less frequent MultiPV, only for strong vs strong, early game, lower depth
                is_strong_vs_strong = (white_mode == "engine_strong" and black_mode == "engine_strong")
                should_consider_alt_move = (
                    is_strong_vs_strong 
                    and moves_played < 6 # Slightly extend opening phase for variety
                    and depth <= 6      # Only use MultiPV for shallower searches
                    and rnd.random() < 0.25 # Reduced probability
                )
                
                if should_consider_alt_move:
                    try:
                        # Removed game=game_id parameter
                        analysis = engine.analyse(board, limit, multipv=2, options=options)
                        if len(analysis) > 1 and 'pv' in analysis[1] and analysis[1]['pv']:
                            move = analysis[1]['pv'][0]
                        else:
                             result = engine.play(board, limit, options=options)
                             move = result.move
                    except chess.engine.EngineError as e:
                        # Log specific engine errors for this game, but continue if possible
                        # print(f"Warning [Game {seed}]: Engine analysis error: {e}. Falling back.")
                        try:
                            result = engine.play(board, limit, options=options)
                            move = result.move
                        except chess.engine.EngineError as e2:
                             # print(f"Error [Game {seed}]: Engine play error after analysis fallback: {e2}. Skipping game.")
                             return None # Critical failure for this game simulation
                
                if move is None:
                    try:
                        # Removed game=game_id parameter
                        result = engine.play(board, limit, options=options)
                        move = result.move
                    except chess.engine.EngineError as e:
                        # print(f"Error [Game {seed}]: Engine play error: {e}. Skipping game.")
                        return None # Critical failure for this game simulation
            
            if move is None: 
                 # print(f"Warning [Game {seed}]: Could not determine a move for {board.fen()}. Skipping game.")
                 return None 
                 
            if move in board.legal_moves:
                board.push(move)
                moves_played += 1
                game_fens.append(board.fen()) # Store FEN string
            else:
                # If an illegal move was somehow generated (e.g., engine bug), discard game
                # print(f"Warning [Game {seed}]: Illegal move {move} generated for FEN {board.fen()}. Discarding game.")
                return None # Discard this game simulation

        # After game, select a random FEN if history exists
        if game_fens:
            return rnd.choice(game_fens)
        else:
            # print(f"Warning [Game {seed}]: Game {white_mode} vs {black_mode} produced no history.")
            return None

    except chess.engine.EngineTerminatedError:
        # print(f"Error [Game {seed}]: Stockfish engine terminated unexpectedly.")
        return None # Indicate failure for this game
    except FileNotFoundError:
        # This should ideally be caught before starting workers, but handle here too
        # print(f"Error [Game {seed}]: Stockfish not found at '{stockfish_path}'.")
        return None
    except Exception as e:
        # Catch any other unexpected error during simulation
        # print(f"Error [Game {seed}]: An unexpected error occurred: {e}")
        return None
    finally:
        # Ensure the engine specific to this process is quit
        if engine:
            engine.quit()

def generate_positions(n=100, stockfish_path="stockfish", workers=None):
    """
    Generate `n` unique chess positions in parallel using multiple Stockfish engines.

    Uses a ProcessPoolExecutor to simulate games concurrently. Implements optimizations
    like canonical FEN keys, reduced search depths, and storing FEN strings.

    Args:
        n (int): The number of unique positions to generate.
        stockfish_path (str): Path to the Stockfish executable.
        workers (int, optional): Number of worker processes. Defaults to cpu_count().

    Returns:
        list[chess.Board]: A list of unique chess board positions.
        
    Raises:
        FileNotFoundError: If the stockfish executable cannot be found *before* starting workers.
        # Other exceptions like EngineTerminatedError are handled per-worker.
    """
    # Check for stockfish existence *before* starting the pool
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            pass # Just check if it starts and quits ok
    except FileNotFoundError:
        print(f"Error: Stockfish executable not found at '{stockfish_path}'.")
        print("Please install Stockfish and provide the correct path.")
        raise
    except chess.engine.EngineError as e:
         print(f"Error: Could not initialize Stockfish engine: {e}")
         raise # Re-raise other engine init errors

    final_boards = []
    seen_keys = set() # Use canonical keys for uniqueness
    
    workers = workers or os.cpu_count()
    
    # Define scenarios inside the function
    scenarios = [
        ("engine_strong", "engine_strong"), ("engine_strong", "engine_weak"),
        ("engine_weak", "engine_strong"), ("engine_weak", "engine_weak"),
        ("engine_strong", "engine_mid"), ("engine_mid", "engine_strong"),
        ("engine_mid", "engine_mid"), ("engine_weak", "engine_mid"),
        ("engine_mid", "engine_weak"), ("engine_strong", "random"),
        ("random", "engine_strong"), ("engine_weak", "random"),
        ("random", "engine_weak"), ("engine_mid", "random"),
        ("random", "engine_mid"), ("random", "random")
    ]

    # Estimate how many games to run - aim for more than n due to failures/duplicates
    # Increase multiplier if duplicates are very common or failures frequent
    num_tasks = n * 4 
    
    tasks_submitted = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Create argument tuples for each task
        args_gen = (
            (i, random.choice(scenarios), stockfish_path) 
            for i in range(num_tasks) # Generate seeds 0 to num_tasks-1
        )
        
        print(f"Generating {n} positions using up to {workers} workers...")
        
        # Use tqdm for progress tracking based on successful positions found
        with tqdm(total=n, unit="position", desc="Generating positions") as pbar:
            # imap_unordered processes tasks as they complete
            for fen_result in pool.map(simulate_game, args_gen):
                if fen_result:
                    try:
                        board = chess.Board(fen_result)
                        key = canon_key(board)
                        
                        if key not in seen_keys:
                            seen_keys.add(key)
                            final_boards.append(board)
                            pbar.update(1)
                            
                            # Check if we have enough positions
                            if len(final_boards) >= n:
                                break # Exit loop once n positions are collected
                    except ValueError:
                        # Handle potential errors if FEN string is invalid (shouldn't happen often)
                        # print(f"Warning: Invalid FEN '{fen_result}' received from worker. Skipping.")
                        pass # Just skip this result

            # If loop finished but not enough boards, check if pool needs shutdown early
            # (Pool context manager handles shutdown, but break above might leave tasks)

    end_time = time.time()
    duration = end_time - start_time
    
    if len(final_boards) < n:
        print(f"\nWarning: Only generated {len(final_boards)} unique positions out of the requested {n} in {duration:.2f}s.")
    else:
        print(f"\nSuccessfully generated {len(final_boards)} unique positions in {duration:.2f}s.")
        
    return final_boards

# Keep the original function signature as a wrapper or replace it
# For simplicity, let's replace the old function name
# def generate_diverse_positions(...): # Old function removed or commented out

if __name__ == '__main__':
    # Example usage: Generate 100 diverse positions using parallel processing
    try:
        # Find stockfish - adjust path if necessary
        # stockfish_executable = "/opt/homebrew/bin/stockfish" # Example macOS
        stockfish_executable = "stockfish" # Assumes in PATH

        # Specify number of workers (e.g., 4 or None for all cores)
        num_workers = os.cpu_count() 
        
        generated_boards = generate_positions(
            n=5000, # Generate more positions for a better test
            stockfish_path=stockfish_executable, 
            workers=num_workers
        )
        
        print(f"Total unique positions generated: {len(generated_boards)}")
        
        # Print the FEN of the first 5 generated positions
        for i, board in enumerate(generated_boards[:5]):
            print(f"Position {i+1}: {canon_key(board)} (Full FEN: {board.fen()})")
            
        # Example analysis
        if generated_boards:
             print(f"\nExample analysis of the first board:")
             print(generated_boards[0])
             print(f"Piece count: {len(generated_boards[0].piece_map())}")
             print(f"Is checkmate? {generated_boards[0].is_checkmate()}")
             print(f"Is stalemate? {generated_boards[0].is_stalemate()}")
             
    except FileNotFoundError:
        # Error printed within generate_diverse_positions_parallel
        print("Execution failed because Stockfish was not found.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        # Print traceback for debugging unexpected errors in the main block
        import traceback
        traceback.print_exc() 