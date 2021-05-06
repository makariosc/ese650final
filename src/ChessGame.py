import chess

#Chess Game class for general use. Needs some work.

class ChessGame(object):

    def __init__(self):
        self.board = chess.Board()
        self.WhiteMove = True

    def get_board(self):
        return self.board

    def legal(self):
        return list(self.board.legal_moves )

    def move(self, move):
        self.WhiteMove = not self.WhiteMove
        self.board.push(move)

        return self.board

    def gameOver(self):
        if(self.board.is_stalemate() ):
            return True, 0

        elif(self.board.is_checkmate() ):
            return True, -(2*self.WhiteMove - 1)

        else:
            return False, 0

        
