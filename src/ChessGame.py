import chess
import mcts
import utils

#Chess Game class for general use. Needs some work.

class ChessGame(object):

    def __init__(self, whiteNN, blackNN):
        self.board = chess.Board()
        self.currPlayer = chess.WHITE

        self.whiteNN = whiteNN
        self.blackNN = blackNN

        self.gameTree = mcts.MCTS(self.board)

        # Training examples
        self.moves = {chess.WHITE : [],
                      chess.BLACK : []}

    def get_board(self):
        return self.board

    def legal(self):
        return list(self.board.legal_moves)

    def selectMove(self, iterations = 10):

        if self.currPlayer == chess.WHITE:
            currNet = self.whiteNN
        else:
            currNet = self.blackNN

        for _ in range(iterations):
            self.gameTree.search(self.gameTree.root, currNet)

        mIdx, pi = self.gameTree.select(1, True)
        theMove = utils.idxToMove(mIdx, self.gameTree.root.state)
        self.gameTree.root = self.gameTree.root.children[mIdx].nextState

        self.move(theMove)
        
        
        showBoard = True # quick flag for showing moves
        
        if showBoard:
            print(theMove)
    
            print("=====")
            print(self.gameTree.root.state)

        # Add the move to our training examples
        self.moves[self.currPlayer].append([utils.makeFeatures(self.board), pi, 0])


    def move(self, move):
        self.currPlayer = not self.currPlayer
        self.board.push(move)

        return self.board

    def gameOver(self):
        if self.board.is_stalemate():
            return True, 0

        elif self.board.is_checkmate():
            # 1 if White, -1 if Black

            #If I understand this correctly, the "turn player" on a checkmate state is the loser. If it's black's turn and checkmate then white wins.
            turnPlay = (self.currPlayer == chess.BLACK)
            return True, 2*turnPlay - 1
        elif self.board.outcome() is not None:
            return True, 0

        else:
            return False, 0

    def gameLoop(self):

        while True:
            self.selectMove()

            done, v = self.gameOver()

            if done:
                print(f"TERMINATED: {self.board.outcome().termination}")

                if v == 0:
                    return self.moves
                elif v == 1:
                    wVal = v
                    bVal = -v
                else:
                    wVal = -v
                    bVal = v

                for sample in self.moves[chess.WHITE]:
                    sample[2] = wVal
                for sample in self.moves[chess.BLACK]:
                    sample[2] = bVal

                return self.moves


        
