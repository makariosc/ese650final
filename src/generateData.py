from alphaZeroNet import ChessNet, train
import utils
import mcts
import torch
import chess
from ChessGame import ChessGame

# best current player plays 25000 games against itself
# uses MCTS to select a move
# at each move, store: game state, search probs from MCTS, who won game (add after game ends)

if __name__=="__main__":
    print('poop')
    # need to load best current model
    # should be saved from torch.save(model.state_dict(), PATH)
    # loaded with:
    # model = modelclass()
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    
    path = 'currentNN.pt'  # whatever the current best model is saved as
    
    model = ChessNet()
    model.load_state_dict(torch.load(path))
    model.eval() # make sure model doesn't change
    
    num_games = 25000
    
    game = ChessGame(model, model)
    data = game.gameLoop()