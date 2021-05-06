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

        self.gameTree = mcts.MCTS()

        # Training examples
        self.moves = {chess.WHITE : [],
                      chess.BLACK : []}

    def get_board(self):
        return self.board

    def legal(self):
        return list(self.board.legal_moves )

    def selectMove(self, iterations = 100):

        if self.currPlayer == chess.WHITE:
            currNet = self.whiteNN
        else:
            currNet = self.blackNN

        for _ in range(iterations):
            self.gameTree.search(self.gameTree.root, currNet)

        mIdx, pi = self.gameTree.select(1, True)
        theMove = utils.idxToMove(mIdx)
        self.gameTree = self.gameTree.children[mIdx].nextState

        self.move(theMove)

        # Add the move to our training examples
        self.moves[chess.currPlayer].append([utils.makeFeatures(self.board), pi, 0])


    def move(self, move):
        self.currPlayer = not self.currPlayer
        self.board.push(move)

        return self.board

    def gameOver(self):
        if self.board.is_stalemate():
            return True, 0

        elif self.board.is_checkmate():
            # 1 if White, -1 if Black
            return True, 2*self.board.winner() - 1

        else:
            return False, 0

    def gameLoop(self):

        while True:
            self.selectMove()

            done, v = self.gameOver()

            if done:
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


        
