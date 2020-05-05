# Marin Pavlinov Marinov
# CSCI 350 Homework 2

from BaseAI_3 import BaseAI
from time import process_time


class PlayerAI(BaseAI):

    def __init__(self):
        # bounds picked as a result of the highest and lowest values found after 25 runs
        self.upperBound = 36000000

        # start time will be used to ensure algorithm is system invariant
        self.startTime = 0

    def getMove(self, grid):

        # start time will be used to ensure algorithm is system invariant and moves do not time out
        self.startTime = process_time()

        # get the move from search algorithm
        move = (self.expectiAlphaBeta(grid))[1]

        return move

    def getHeuristics(self,grid):
        """ Returns weighted sum of the 5 heuristics """
        return self.monotonicPatternHeuristic(grid)

#######################################
## Heuristics
#######################################

    def snakePatternHeuristic(self,grid):
        """ Snake weighted matrix pattern heuristic """
        score = 0

        # I want the tiles to keep the highest number in the lower left corner 
        score += grid.getCellValue((0, 0)) * 0
        score += grid.getCellValue((0, 1)) * 1
        score += grid.getCellValue((0, 2)) * 4
        score += grid.getCellValue((0, 3)) * 9
        score += grid.getCellValue((1, 0)) * 49
        score += grid.getCellValue((1, 1)) * 36
        score += grid.getCellValue((1, 2)) * 25
        score += grid.getCellValue((1, 3)) * 16
        score += grid.getCellValue((2, 0)) * 64
        score += grid.getCellValue((2, 1)) * 81
        score += grid.getCellValue((2, 2)) * 100
        score += grid.getCellValue((2, 3)) * 121
        score += grid.getCellValue((3, 0)) * 225 # corner
        score += grid.getCellValue((3, 1)) * 196
        score += grid.getCellValue((3, 2)) * 169
        score += grid.getCellValue((3, 3)) * 144

        return score

    def monotonicPatternHeuristic(self,grid):
        """ Heuristic that tries to ensure that the tiles follow a pattern """
        score = 0

        # I want the tiles to keep the highest number in the upper right corner 
        score += grid.getCellValue((0, 0)) * 81
        score += grid.getCellValue((0, 1)) * 49
        score += grid.getCellValue((0, 2)) * 0
        score += grid.getCellValue((0, 3)) * 0
        score += grid.getCellValue((1, 0)) * 121
        score += grid.getCellValue((1, 1)) * 81
        score += grid.getCellValue((1, 2)) * 49
        score += grid.getCellValue((1, 3)) * 0
        score += grid.getCellValue((2, 0)) * 169
        score += grid.getCellValue((2, 1)) * 121
        score += grid.getCellValue((2, 2)) * 81
        score += grid.getCellValue((2, 3)) * 49
        score += grid.getCellValue((3, 0)) * 225 # corner
        score += grid.getCellValue((3, 1)) * 169
        score += grid.getCellValue((3, 2)) * 121
        score += grid.getCellValue((3, 3)) * 81

        return score

#######################################
## Algorithms
#######################################

    def expectimax(self, grid, isMaxPlayer=True, depth=4):
        """ performs expectimax algorithm to determine best move """

        playerMoveset = grid.getAvailableMoves()

        # base case
        if depth == 0 or process_time() - self.startTime > 0.15 or not playerMoveset:
            return self.getHeuristics(grid), 0

        if isMaxPlayer:
            maxScore = -1 * float("inf")
            for move, gridCopy in playerMoveset:
                currScore, currMove = self.expectimax(
                    gridCopy, False, depth - 1)

                if currScore > maxScore:
                    bestMove = move
                    maxScore = currScore

            return maxScore, bestMove
        else:
            # Ai moveset consists of it putting a 2 or a 4 in any of the availabe cells left
            aiMoveset = [(n,tile) for n in [2,4] for tile in grid.getAvailableCells()]
            avg = 0
            twos = 0
            fours = 0
            for val, tile in aiMoveset:
                gridCopy = grid.clone()

                if tile and gridCopy.canInsert(tile):
                   gridCopy.setCellValue(tile, val)

                   currScore, currMove = self.expectimax(
                       gridCopy, True, depth - 1)
                   
                   # weight based on probability (90% for 2 and 10% for 4)
                   if val == 2:
                       avg += currScore * 0.9
                       twos += 1
                   else:
                       avg += currScore * 0.1 
                       fours += 1

            # return weighted average 
            return avg / (0.9 * twos + 0.1 * fours), 0

    def alphaBeta(self, grid, isMaxPlayer=True, depth=5, alpha=-1 * float("inf"), beta=float("inf")):
        """ Performs minimax with alpha beta pruning """

        # Order of moves (DOWN, LEFT, RIGHT, UP)
        playerMoveset = grid.getAvailableMoves([1,2,3,0])

        # base case
        if depth == 0 or process_time() - self.startTime > 0.15 or not playerMoveset:
            return self.getHeuristics(grid), 0

        if isMaxPlayer:
            maxScore = -1 * float("inf")
            for move, gridCopy in playerMoveset:
                currScore, currMove = self.alphaBeta(
                    gridCopy, False, depth - 1, alpha, beta)

                if currScore > maxScore:
                    bestMove = move
                    maxScore = currScore

                # alpha-beta pruning check step
                alpha = max(alpha, currScore)
                if beta <= alpha:
                    break

            return maxScore, bestMove
        else:
            aiMoveset = [(n,tile) for n in [2,4] for tile in grid.getAvailableCells()]
            minScore = float("inf")
            for val, tile in aiMoveset:
                gridCopy = grid.clone()

                if tile and gridCopy.canInsert(tile):
                   gridCopy.setCellValue(tile, val) 

                   currScore, currMove = self.alphaBeta(
                       gridCopy, True, depth - 1, alpha, beta)

                   if currScore < minScore:
                       minScore = currScore

                   # alpha-beta prunning check step
                   beta = min(beta, currScore)
                   if beta <= alpha:
                       break

            return minScore, 0

    # The parameter node takes the following values (I found this to perform better than True and False for some reason)
    # 1 = max node
    # 2 = chance node
    def expectiAlphaBeta(self, grid, node=1, depth=3, alpha=-1 * float("inf"), beta=float("inf")):
        """ Performs expectimax with alpha beta pruning """

        # Order of moves (DOWN, LEFT, RIGHT, UP)
        playerMoveset = grid.getAvailableMoves([1,2,3,0])

        # base case
        if depth == 0 or process_time() - self.startTime > 0.15 or not playerMoveset:
            return self.getHeuristics(grid), 0

        if node == 1:
            maxScore = -1 * float("inf")
            for move, gridCopy in playerMoveset:
                currScore, currMove = self.expectiAlphaBeta(
                    gridCopy, 2, depth - 1, alpha, beta)

                if currScore > maxScore:
                    alpha = currScore
                    bestMove = move
                    maxScore = currScore

            return maxScore, bestMove
        else:
            # Ai moveset consists of it putting a 2 or a 4 in any of the availabe cells left
            aiMoveset = [(n,tile) for n in [2,4] for tile in grid.getAvailableCells()]
            avg = 0
            currWeight = 0
            for val, tile in aiMoveset:
                gridCopy = grid.clone()

                if tile and gridCopy.canInsert(tile):
                    gridCopy.setCellValue(tile, val)

                    currScore = (self.expectiAlphaBeta(
                       gridCopy, 1, depth - 1, -1 * float("inf"), beta))[0]

                    # weight based on probability (90% for 2 and 10% for 4)
                    if val == 2:
                        avg += currScore * 0.9
                        currWeight += 0.9
                    else:
                        avg += currScore * 0.1
                        currWeight += 0.1

                    # alpha-beta pruning check step
                    beta = min(beta, currScore)
                    if beta <= alpha:
                        break

            # return weighted average
            return avg / currWeight, 0
