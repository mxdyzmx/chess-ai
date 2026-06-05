"""
Feature extraction module for chess position evaluation.
Must match the C++ engine's feature extraction exactly.

Feature layout (391 total):
  0-63:   pawn piece-square (A1=0, H1=7, A2=8, ..., H8=63)
  64-127: knight piece-square
  128-191: bishop piece-square
  192-255: rook piece-square
  256-319: queen piece-square
  320-383: king piece-square
  384: mobility (pseudo-legal moves / 50, capped to [0, 1])
  385: white isolated pawn count
  386: black isolated pawn count
  387: white passed pawn count
  388: black passed pawn count
  389: white king safety damage (pawn shield weakness, 0-4)
  390: black king safety damage (pawn shield weakness, 0-4)
"""

import chess
import chess.pgn
import numpy as np
from typing import List, Optional

NUM_FEATURES = 391

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def square_to_index(sq: int) -> int:
    """Convert python-chess square (0-63, A1=0) to our index (same mapping)."""
    return sq


def extract_features(board: chess.Board) -> np.ndarray:
    """Extract 391 features from a chess position.

    Args:
        board: A python-chess Board object.

    Returns:
        numpy array of shape (391,) with float32 features.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # 1. Piece-square features (0-383): +1 for white, -1 for black
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        pt = piece.piece_type
        color = piece.color
        sq_idx = square_to_index(square)
        feat_idx = PIECE_TO_INDEX[pt] * 64 + sq_idx
        features[feat_idx] = 1.0 if color == chess.WHITE else -1.0

    # 2. Mobility (384): number of pseudo-legal moves / 50, capped at 1.0
    num_moves = len(list(board.generate_pseudo_legal_moves()))
    features[384] = min(num_moves / 50.0, 1.0)

    # 3. Pawn structure features (385-388)
    w_pawns = board.pieces(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces(chess.PAWN, chess.BLACK)

    features[385] = float(count_isolated_pawns(w_pawns))
    features[386] = float(count_isolated_pawns(b_pawns))
    features[387] = float(count_passed_pawns(w_pawns, b_pawns, chess.WHITE))
    features[388] = float(count_passed_pawns(b_pawns, w_pawns, chess.BLACK))

    # 4. King safety (389-390)
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    features[389] = float(king_safety_damage(w_king, w_pawns, chess.WHITE))
    features[390] = float(king_safety_damage(b_king, b_pawns, chess.BLACK))

    return features


def count_isolated_pawns(pawns: chess.SquareSet) -> int:
    """Count number of isolated pawns in a pawn bitmask.

    A pawn is isolated if there are no friendly pawns on adjacent files.
    """
    count = 0
    for sq in pawns:
        file = chess.square_file(sq)
        has_left = file > 0 and bool(pawns & chess.BB_FILES[file - 1])
        has_right = file < 7 and bool(pawns & chess.BB_FILES[file + 1])
        if not has_left and not has_right:
            count += 1
    return count


def count_passed_pawns(pawns: chess.SquareSet, enemy_pawns: chess.SquareSet, color: chess.Color) -> int:
    """Count number of passed pawns.

    A pawn is passed if there are no enemy pawns on the same or adjacent files
    ahead of it.
    """
    count = 0
    for sq in pawns:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        passed = True
        for df in (-1, 0, 1):
            f = file + df
            if f < 0 or f > 7:
                continue
            file_mask = chess.BB_FILES[f]
            if color == chess.WHITE:
                # Squares ahead (higher ranks)
                ahead_mask = file_mask & (~((1 << (sq + 1)) - 1))
            else:
                # Squares ahead (lower ranks)
                ahead_mask = file_mask & ((1 << sq) - 1)
            if enemy_pawns & ahead_mask:
                passed = False
                break
        if passed:
            count += 1
    return count


def king_safety_damage(king_sq: Optional[int], pawns: chess.SquareSet, color: chess.Color) -> int:
    """Evaluate king safety based on pawn shield integrity.

    Checks the squares in front of the king on the same and adjacent files.
    Returns 0 (intact shield) to 4 (badly damaged).
    """
    if king_sq is None:
        return 4

    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)

    # Compute the pawn shield mask (squares in front of the king)
    # Two ranks ahead, three files wide
    shield_squares = []
    for df in (-1, 0, 1):
        f = king_file + df
        if f < 0 or f > 7:
            continue
        if color == chess.WHITE:
            for dr in (1, 2):
                r = king_rank + dr
                if r > 7:
                    continue
                shield_squares.append(chess.square(f, r))
        else:
            for dr in (-1, -2):
                r = king_rank + dr
                if r < 0:
                    continue
                shield_squares.append(chess.square(f, r))

    # Count how many of the shield squares have pawns
    shield_pawns = sum(1 for sq in shield_squares if sq in pawns)
    missing = len(shield_squares) - shield_pawns

    # Cap damage at 4 (max feature value)
    return min(missing, 4)


def features_from_fen(fen: str) -> np.ndarray:
    """Extract features from a FEN string."""
    board = chess.Board(fen)
    return extract_features(board)
