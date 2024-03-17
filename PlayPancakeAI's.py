import chess
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import *

def generate_all_possible_uci_moves():
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promotions = ['q', 'r', 'b', 'n']  # Queen, Rook, Bishop, Knight
    all_moves = []

    for start_letter in letters:
        for start_number in numbers:
            for end_letter in letters:
                for end_number in numbers:
                    for promotion in promotions:
                        # Generating standard moves
                        move = f"{start_letter}{start_number}{end_letter}{end_number}"
                        all_moves.append(move)
                        # Generating promotion moves
                        promotion_move = f"{start_letter}{start_number}{end_letter}{end_number}{promotion}"
                        all_moves.append(promotion_move)

    # Removing duplicates if any
    all_moves = list(set(all_moves))
    return all_moves

class NN(nn.Module):
    def __init__(self, num_moves=4672):
        """
        Initializes the modified AlphaZero-like neural network.
        
        Parameters:
        - num_moves: Number of possible moves from any given board state.
        """
        super(NN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Adjusted for a flat input of 71, with a reshaped board of 8x8 and additional features
        self.fc_input_size = 128 * 8 * 8 + 5  # +5 for the extra features (4 for castling, 1 for turn)
        
        # Additional FC layer for integrating extra features
        self.fc_extra = nn.Linear(71, self.fc_input_size)
        
        # Policy head
        self.policy_fc1 = nn.Linear(self.fc_input_size, 512)
        self.policy_fc2 = nn.Linear(512, num_moves)
        
        # Value head
        self.value_fc1 = nn.Linear(self.fc_input_size, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        # Correctly slice the input tensor to separate the board state (first 64 elements)
        # from the extra features (remaining elements)
        x = x.unsqueeze(0)
        board_state = x[:, :64].view(-1, 1, 8, 8)  # Reshape for convolutional layers
        extra_features = x[:, 64:]  # Assuming there are 5 extra features

        # Convolutional layers
        x = F.relu(self.conv1(board_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for FC layers
        x = x.view(-1, 128 * 8 * 8)
        # Concatenate the extra features after convolutional processing
        x = torch.cat((x, extra_features), dim=1)

        # Additional FC layer for processing
        self.fc_extra = nn.Linear(8197, self.fc_input_size)  # Adjusted input size to match the actual size of `x`
        
        # Policy head
        policy = F.relu(self.policy_fc1(x))
        probs = F.softmax(self.policy_fc2(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))
        
        return probs, value
    
    @staticmethod
    def encodeBoard(board):
        """
        Encodes the board state into a tensor.
        
        Parameters:
        - board: A chess.Board object.
        
        Returns:
        - A torch.Tensor representing the encoded board state.
        """
        # Initialize an empty tensor for the board state
        encoded = torch.zeros(8 * 8 + 4 + 1)  # 64 squares + 4 castling rights + 1 turn
        
        # Map pieces to integers
        piece_to_int = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                encoded[i] = piece_to_int.get(piece.symbol(), 0)
        
        # Encode castling rights
        encoded[64] = int(board.has_queenside_castling_rights(chess.WHITE))
        encoded[65] = int(board.has_kingside_castling_rights(chess.WHITE))
        encoded[66] = int(board.has_queenside_castling_rights(chess.BLACK))
        encoded[67] = int(board.has_kingside_castling_rights(chess.BLACK))
        
        # Encode the turn
        encoded[68] = 0 if board.turn == chess.WHITE else 1 #0 if it's white's turn, 1 if black's
        
        return encoded

def treeSearch(board, times):
    original_fen = board.fen()  # Save the original board state
    color = board.turn
    result = 0
    legalMoves = list(board.legal_moves)
    
    for i in range(times):
        board.set_fen(original_fen)  # Reset the board to its original state
        if len(legalMoves) > 0:
            board.push(legalMoves[i % len(legalMoves)]) #Expansion
        else: #Game Over
            break
        while not board.is_game_over():
            board.push(random.choice(list(board.legal_moves)))
        if (board.result() == "1-0" and color == chess.WHITE) or (board.result() == "0-1" and color == chess.BLACK):
            result += 1
        elif (board.result() == "0-1" and color == chess.WHITE) or (board.result() == "1-0" and color == chess.BLACK):
            result -= 1
        else:
            result = 0
    board.set_fen(original_fen)
    return result / times

def softmaxToDistribution(board, softmax, iterations):
    # Assuming softmax is a list of probabilities corresponding to some ordering of moves
    legal_moves = list(board.legal_moves)
    softmax_probabilities = {move.uci(): prob for move, prob in zip(legal_moves, softmax)}

    # This example simply retains the softmax probability for legal moves,
    # You might want to adjust this logic depending on how you associate moves with softmax probabilities
    legal_softmax = [softmax_probabilities.get(move.uci(), 0) for move in legal_moves]

    # Assuming the intent is to scale or otherwise process these probabilities:
    # (Note: This is placeholder logic. Adjust according to your actual intent.)
    distributed_softmax = [prob * iterations for prob in legal_softmax]

    return distributed_softmax

def decideMove(board, model):
    moveScores = []
    originalBoard = board.fen()
    legal_moves = list(board.legal_moves)
    encodedBoard = NN.encodeBoard(board)#.unsqueeze(0)  # Add a batch dimension
    #print(encodedBoard)

    # Forward pass through the model to get probabilities and board value
    p, v = model(encodedBoard)
    p = p.squeeze().detach().numpy()  # Convert tensor to numpy array for easier handling
    v = v.item()  # Convert tensor to scalar

    # Generate and use move_to_index mapping inside the decideMove function
    all_possible_uci_moves = generate_all_possible_uci_moves()
    move_to_index = {move: i for i, move in enumerate(all_possible_uci_moves)}

    # Ensure each move is legal and has a corresponding probability
    for move in legal_moves:
        move_uci = move.uci()
        if move_uci in move_to_index and move_to_index[move_uci] < len(p):
            index = move_to_index[move_uci]
            prob = p[index]  # Get probability from model output
        else:
            prob = 0  # Assign a zero probability if the move is not recognized by the model

        # Evaluate the move's score using treeSearch
        board.push(move)  # Make the move on the board
        score = treeSearch(board, max(int(prob * 100 * abs(v)), 1))  # Adjust treeSearch as needed
        moveScores.append(score)
        board.set_fen(originalBoard)

    # Select the best move based on scores
    if moveScores:
        bestScore = max(moveScores)
        bestMoves = [move for move, score in zip(legal_moves, moveScores) if score == bestScore]
        bestMove = random.choice(bestMoves) if bestMoves else None
    else:
        bestMove = None

    return bestMove



    
    
    
        
model = torch.load("PancakeAI(Not Very Good).pth")
#PVAI
board = chess.Board()
while not board.is_game_over():
    bestMove = decideMove(board, model)
    print("Best move:", board.san(bestMove))
    board.push(bestMove)
    print(board)
    if board.is_game_over():
        break
    playerMove = input("Play a move: ")
    while True:
        try:
            move = board.parse_san(playerMove)
            if move in board.legal_moves:
                board.push(move)
                print(board)
                break
            else:
                print("Not a legal move! Input another")
        except ValueError:
            print("Not a legal move! Input another")
            playerMove = input("Play a move: ")
