# Final AI Project

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
        return self.snakePatternHeuristic(grid) * 3 + self.mergeHeuristic(grid) * 2 + self.openHeuristic(grid)

#######################################
## Heuristics
#######################################

    def snakePatternHeuristic(self,grid):
        """ Snake weighted matrix pattern heuristic """
        score = 0

        # I want the tiles to keep the highest number in the lower left corner 
        score += grid.getCellValue((0, 0)) * 1
        score += grid.getCellValue((0, 1)) * 4
        score += grid.getCellValue((0, 2)) * (4 ** 2)
        score += grid.getCellValue((0, 3)) * (4 ** 3)
        score += grid.getCellValue((1, 0)) * (4 ** 7)
        score += grid.getCellValue((1, 1)) * (4 ** 6)
        score += grid.getCellValue((1, 2)) * (4 ** 5)
        score += grid.getCellValue((1, 3)) * (4 ** 4)
        score += grid.getCellValue((2, 0)) * (4 ** 8)
        score += grid.getCellValue((2, 1)) * (4 ** 9)
        score += grid.getCellValue((2, 2)) * (4 ** 10)
        score += grid.getCellValue((2, 3)) * (4 ** 11)
        score += grid.getCellValue((3, 0)) * (4 ** 15) # corner
        score += grid.getCellValue((3, 1)) * (4 ** 14)
        score += grid.getCellValue((3, 2)) * (4 ** 13)
        score += grid.getCellValue((3, 3)) * (4 ** 12)

        return score

    def monotonicPatternHeuristic(self,grid):
        """ Heuristic that tries to ensure that the tiles follow a  monotonic pattern """
        score = 0

        # I want the tiles to keep the highest number in the lower left corner 
        score += grid.getCellValue((0, 0)) * (4 ** 3)
        score += grid.getCellValue((0, 1)) * (4 ** 2)
        score += grid.getCellValue((0, 2)) * 4
        score += grid.getCellValue((0, 3)) * 1
        score += grid.getCellValue((1, 0)) * (4 ** 4)
        score += grid.getCellValue((1, 1)) * (4 ** 3)
        score += grid.getCellValue((1, 2)) * (4 ** 2)
        score += grid.getCellValue((1, 3)) * 4
        score += grid.getCellValue((2, 0)) * (4 ** 5)
        score += grid.getCellValue((2, 1)) * (4 ** 4)
        score += grid.getCellValue((2, 2)) * (4 ** 3)
        score += grid.getCellValue((2, 3)) * (4 ** 2)
        score += grid.getCellValue((3, 0)) * (4 ** 6) # corner
        score += grid.getCellValue((3, 1)) * (4 ** 5)
        score += grid.getCellValue((3, 2)) * (4 ** 4)
        score += grid.getCellValue((3, 3)) * (4 ** 3)

        return score

    def mergeHeuristic(self,grid):
        """ Heuristics that rewards for the same values next to each other """
        score = 0
        for i in range(4):
            for j in range(4):
                curr = grid.getCellValue((i,j))
                neighborUp = grid.getCellValue((i - 1,j))
                neighborDown = grid.getCellValue((i + 1,j))
                neighborLeft = grid.getCellValue((i,j + 1))
                neighborRight = grid.getCellValue((i,j - 1))

                if curr == neighborUp or curr == neighborDown or curr == neighborLeft or curr == neighborRight:
                    score += curr * (4 ** 8)

        return score

    def openHeuristic(self,grid):
        """ Heuristic that grants bonuses for the number of available tiles"""
        return len(grid.getAvailableCells()) * (4 ** 10)

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
    def expectiAlphaBeta(self, grid, node=1, depth=5, alpha=-1 * float("inf"), beta=float("inf")):
        """ Performs expectimax with alpha beta pruning """

        # Order of moves (DOWN, LEFT, RIGHT, UP)
        playerMoveset = grid.getAvailableMoves([1,2,3,0])

        # base case
        if depth == 0 or not playerMoveset:
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
