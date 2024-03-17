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
    encodedBoard = encodedBoard.to("cuda")
    #print(encodedBoard)

    # Forward pass through the model to get probabilities and board value
    p, v = model(encodedBoard)
    p = p.squeeze().detach().cpu().numpy()  # Convert tensor to numpy array for easier handling
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

def adjust_and_normalize_probs(legal_moves, softmax_probs, all_possible_moves):
    # Initialize a dictionary to hold the adjusted probabilities for legal moves
    adjusted_probs = {move: 0 for move in all_possible_moves}
    
    # Populate the dictionary with softmax probabilities for legal moves
    for move in legal_moves:
        move_uci = move.uci()
        if move_uci in softmax_probs:
            adjusted_probs[move_uci] = softmax_probs[move_uci]
    
    # Normalize the probabilities so they sum to 1
    total_prob = sum(adjusted_probs.values())
    if total_prob > 0:  # Avoid division by zero
        normalized_probs = {move: prob / total_prob for move, prob in adjusted_probs.items()}
    else:
        normalized_probs = adjusted_probs  # In case all probs are 0, which should be handled differently
    
    return normalized_probs



#Self-Training
trainingBoard = chess.Board()
model = NN(4672)
if torch.cuda.is_available():
    model.to("cuda")
    device = "cuda"
else:
    model.to("cpu")
    device = "cpu"
policyCriterion = nn.CrossEntropyLoss()
valueCriterion = nn.MSELoss()
steps = 2
universalMoves = generate_all_possible_uci_moves()
model.train()  # Set the model to training mode
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer, feel free to adjust the learning rate
for i in range(steps): #A step is a game, and then a backpropagation
    states = []
    p = [] #Policy vectors
    v = [] #Value scalars
    pi = [] #Actual probability distribution
    z = [] #Actual Outcome (to v)
    
    while not trainingBoard.is_game_over == True:
        move = decideMove(trainingBoard, model)
        legalMoves = list(trainingBoard.legal_moves)
        state = model.encodeBoard(trainingBoard).to("cuda")
        states.append(state)
        pvalue, vvalue = model.forward(state)
        p.append(pvalue)
        v.append(vvalue)
        # print(pvalue)
        # print(pvalue.type())
        # Ensure pvalue is squeezed to remove the batch dimension
        pvalue_squeezed = pvalue.squeeze()

        # Convert the squeezed tensor to a numpy array
        pvalue_numpy = pvalue_squeezed.detach().cpu().numpy()  # Ensure detachment if gradients are not needed
        # Map each UCI move to its corresponding softmax probability
        softmax_probs_dict = {universalMoves[i]: prob for i, prob in enumerate(pvalue_numpy)}
        # Adjust and normalize the softmax probabilities for legal moves
        pi_adjusted = adjust_and_normalize_probs(legalMoves, softmax_probs_dict, universalMoves)



        # # Ensure pvalue is squeezed to remove the batch dimension, which might be unnecessary here
        # pvalue_squeezed = pvalue.squeeze()  # This should now be a 1D tensor with 4672 elements
        # print(pvalue_squeezed)
        # # Now, you can safely iterate over pvalue_squeezed and map each probability to the corresponding move
        # softmax_probs_dict = {move: pvalue_squeezed[i].item() for i, move in enumerate(universalMoves)}
        # Now, you can pass this dictionary to adjust_and_normalize_probs
        pi.append(pi_adjusted)
        print(trainingBoard.san(move))
        trainingBoard.push(move)
    result = trainingBoard.result()
    if result == "1-0":
        outcome = 1  # White wins
    elif result == "0-1":
        outcome = -1  # Black wins
    else:
        outcome = 0  # Draw

    # Assuming states[] holds the board states in order of play
    for state in states:
        if state.turn == chess.WHITE:
            z.append(outcome)
        else:  # If it's Black's turn, invert the outcome since -1 is a win for Black
            z.append(-outcome)
            
    #TIME TO LEARN
    # Convert your accumulated data into tensors
    # Assuming `states`, `pi`, and `z` are lists of tensors
    # Assuming c is your regularization strength
    c = 0.01

    # Compute the L2 norm squared of all parameters
    l2_norm_squared = sum(p.pow(2.0).sum() for p in model.parameters())

    # Compute the regularization loss
    reg_loss = c * l2_norm_squared
    state = model.encodeBoard(trainingBoard)
    pi_tensor = torch.stack(pi)
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)  # Assuming z is a list of scalar outcomes
    if torch.cuda.is_available():
        states_tensor = states_tensor.to("cuda")
        pi_tensor = pi_tensor.to("cuda")
        z_tensor = z_tensor.to("cuda")

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    policy_output, value_output = model(states_tensor)

    # Compute loss
    policy_loss = policyCriterion(policy_output, pi_tensor)
    value_loss = valueCriterion(value_output, z_tensor)

    total_loss = policy_loss + value_loss + reg_loss   # Combine losses, you can also weigh them differently

    # Backward pass
    total_loss.backward()

    # Update model parameters
    optimizer.step()
    
#Finished training, so now, we save the result:
torch.save(model, 'PancakeAI(Not Very Good).pth')
