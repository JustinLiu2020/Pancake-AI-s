import chess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import math
from math import *

class Eval(nn.Module):
    def __init__(self):
        super(Eval, self).__init__()
        # Define the architecture of the network
        self.layer1 = nn.Linear(69, 128)  # First hidden layer with 128 units
        self.layer2 = nn.Linear(128, 512) # Second hidden layer with 256 units
        self.layer3 = nn.Linear(512, 128) # Third hidden layer with 128 units
        self.layer4 = nn.Linear(128, 512)
        self.output_layer = nn.Linear(512, 1) # Output layer with 1 unit
        
        self.relu = nn.ReLU() # ReLU activation function
        
    def forward(self, x):
        # Define the forward pass
        x = self.relu(self.layer1(x)) # Activation function after first layer
        x = self.relu(self.layer2(x)) # Activation function after second layer
        x = self.relu(self.layer3(x)) # Activation function after third layer
        x = self.relu(self.layer4(x))
        x = torch.sigmoid(self.output_layer(x)) # Sigmoid activation at the output layer
        return x

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

# def minimax_ab_time_limit(model, node, depth, alpha, beta, maximizing_player, end_time):
#     if time.time() > end_time or node.is_game_over():
#         return evaluate_board(model, encodeBoard(node))
    
#     if maximizing_player:
#         max_eval = float('-inf')
#         for move in node.legal_moves:
#             node.push(move)
#             eval = minimax_ab_time_limit(model, node, depth-1, alpha, beta, False, end_time)
#             node.pop()
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break
#         return max_eval
#     else:
#         min_eval = float('inf')
#         for move in node.legal_moves:
#             node.push(move)
#             eval = minimax_ab_time_limit(model, node, depth-1, alpha, beta, True, end_time)
#             node.pop()
#             min_eval = min(min_eval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break
#         return min_eval

# # def iterative_deepening(board, model, time_limit):
# #     end_time = time.time() + time_limit
# #     depth = 1
# #     best_eval = float('-inf')
# #     best_move = None
# #     while time.time() < end_time:
# #         for move in board.legal_moves:
# #             board.push(move)
# #             eval = minimax_ab_time_limit(model, board, depth, float('-inf'), float('inf'), False, end_time)
# #             board.pop()
# #             if eval > best_eval:
# #                 best_eval = eval
# #                 best_move = move
# #         depth += 1
# #     return best_move

    
def evaluate_board(model, board):
    encodedBoard = encodeBoard(board)  # Get the encoded board representation
    # Convert encodedBoard to a tensor with the appropriate shape. Assume model expects a single feature dimension.
    encodedBoard_tensor = torch.tensor([encodedBoard], dtype=torch.float).unsqueeze(0)  # Adds batch and feature dimensions
    eval = model.forward(encodedBoard_tensor)  # Pass the tensor to the model
    return eval  # Convert the tensor output back to a Python scalar

def decideMove(board, model, asdf):
    legalMoves = list(board.legal_moves)
    scores = []
    for i in range(len(legalMoves)):
        #Eval
        board.push(legalMoves[i])
        encodedBoard = encodeBoard(board)
        scores.append(float(model.forward(encodedBoard)) - random.uniform(0, 0.1)) #Exploration
        board.pop()
    bestScore = max(scores)
    bestIndex = scores.index(bestScore)
    return legalMoves[bestIndex]
        
        


def selfPlay(model, optimizer, iterations=10):
    board = chess.Board()
    states = []
    prev_loss = None  # Initialize the previous loss variable

    for i in range(iterations):
        board.reset()  # Resets to the initial board setup
        while not board.is_game_over() and len(board.move_stack) < 100: #50 move games only!!!
            state = encodeBoard(board)
            states.append(state)
            bestMove = decideMove(board, model, 5)
            print(board.san(bestMove))
            board.push(bestMove)

        # Determine the game result
        result = board.result()
        if result == "1-0":
            result_mapped = 1  # White wins
        elif result == "0-1":
            result_mapped = 0  # Black wins
        else:
            result_mapped = 0.5  # Draw/"Timeout"

        # Convert states to a tensor
        states_tensor = torch.stack(states)

        # Predict using the model
        optimizer.zero_grad()  # Zero the gradients
        predictions = model(states_tensor)

        # Convert results to a tensor and adjust dimensions
        results_tensor = torch.FloatTensor([result_mapped] * len(states)).view(-1, 1)

        # Calculate loss
        loss = F.binary_cross_entropy(predictions, results_tensor)

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Print the losses
        if prev_loss is not None:  # This will skip printing on the very first iteration
            print(f"Iteration {i+1}, Prev Loss: {prev_loss.item()}, Current Loss: {loss.item()}")
        else:
            print(f"Iteration {i+1}, Current Loss: {loss.item()}")

        prev_loss = loss  # Update the previous loss

        torch.save(model, "AttemptNN.pth")
        
            
     
model = Eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)
selfPlay(model, optimizer, 1000)