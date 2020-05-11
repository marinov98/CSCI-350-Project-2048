'''
Aryan Bhatt
CSCI 350
HW 2 (Programming)
'''

import random
import math
import time
from BaseAI_3 import BaseAI
from PlayerAI_3 import PlayerAI as Marinov

class PlayerAI(BaseAI):

    def __init__(self, weights = [5,2,1,4,0.1], memo_dic = {}):
        #store previously computed stats to reduce redundant computation
        self.memo = memo_dic
        self.timed_out = False
        #self.MemoCalls = 0
        #self.totalCalls = 0
        #weights for the heuristic
        self.weights = list(weights)
        #time limit to make sure we don't use too much time
        self.time_limit = 1
        #upper bound on heuristic function for alpha-beta pruning (only for expectimax)
        self.UPPER_BOUND = 999
        self.max_heur = -float('inf')

    def getMove(self, grid):
        #initialize time
        '''
        if (len(self.memo)%35 == 0):
            print(len(self.memo))
            print("totalCalls: ", self.totalCalls)
            print("MemoCalls: ", self.MemoCalls)
            print("ratio: ", self.MemoCalls/self.totalCalls if self.totalCalls!=0 else "totalCalls=0")
        '''
        self.timer = time.process_time()
        return self.iterative_deepening_expectimax(grid,1,4)
      # return (Marinov.expectiAlphaBeta(Marinov,grid))[1]
        #return self.iterative_deepening_minimax(grid,1,4)
        #return self.minimax(grid, 3, True, -float('inf'), float('inf'))[1]
        #return self.expectimax(grid, 3, True, -float('inf'), float('inf'))[1]

#######################################
## Algorithms
#######################################

    #iterative deepening search with minimax
    def iterative_deepening_minimax(self, grid, start, end):
        #base case, in case all runs time out
        answer = random.choice([0,2])
        #iterative deepening
        for i in range(start, end):
            self.timed_out = False
            #call minimax with depth cut off at i
            new_answer = self.minimax(grid, i, True, -float('inf'), float('inf'))[1]
            if not self.timed_out:
                answer = new_answer
            else:
                break
        return answer

    #iterative deepening search with expectimax
    def iterative_deepening_expectimax(self, grid, start, end):
        #base case, in case all runs time out
        answer = random.choice([0,2])
        #iterative deepening
        for i in range(start, end):
            self.timed_out = False
            #call expectimax with depth cut off at i
            new_answer = self.expectimax(grid, i, True, -float('inf'), float('inf'))[1]
            if not self.timed_out:
                answer = new_answer
            else:
                break
        return answer

    #minimax with alpha-beta pruning
    def minimax(self, grid, movesLeft, playerTurn, alpha, beta):
        #check if this state has been computed to at least the required depth
        for i in range(movesLeft, 10):
            if (tuple(tuple(row) for row in grid.map), i, playerTurn) in self.memo:
                return self.memo[(tuple(tuple(row) for row in grid.map), i, playerTurn)]

        if playerTurn:
            #player moveset
            moveset = grid.getAvailableMoves([0,2,1,3])
        else:
            #computer moveset consists of placing 2 or 4 into an empty cell
            moveset = [(i,j) for i in [2,4] for j in grid.getAvailableCells()]
        #if not enough time left, cut off recursion
        if (time.process_time()-self.timer > self.time_limit):
            self.timed_out = True
            movesLeft = 0
        #if no more moves or at max depth, cut off recursion
        if movesLeft == 0 or len(moveset) == 0:
            return (self.heuristic(grid), 0)

        #player turn
        if playerTurn:
            #default move is UP (0)
            maxMove = 0
            for move in moveset:
                #recursive call to evaluate child node
                newVal = self.minimax(move[1], movesLeft-1, False, alpha, beta)[0]
                #update max child value
                if newVal > alpha:
                    alpha = newVal
                    maxMove = move[0]
                #alpha-beta pruning
                if alpha >= beta:
                    break
            #store computed result
            self.memo[(tuple(tuple(row) for row in grid.map), movesLeft, playerTurn)] = (alpha, maxMove)
            for i in range(movesLeft):
                if (tuple(tuple(row) for row in grid.map), i, playerTurn) in self.memo:
                    self.memo[(tuple(tuple(row) for row in grid.map), i, playerTurn)] = (alpha, maxMove)
            return (alpha, maxMove)
        #computer turn
        else:
            for move in moveset:
                #place tile into cell to create child state
                newGrid = grid.clone()
                newGrid.setCellValue(move[1], move[0])
                #evaluate child node
                nodeValue = self.minimax(newGrid, movesLeft-1, True, alpha, beta)[0]
                #update min child value
                beta = min(beta, nodeValue)
                #alpha-beta pruning
                if alpha >= beta:
                    break
            #store computed result
            self.memo[(tuple(tuple(row) for row in grid.map), movesLeft, playerTurn)] = (beta, 0)
            for i in range(movesLeft):
                if (tuple(tuple(row) for row in grid.map), i, playerTurn) in self.memo:
                    self.memo[(tuple(tuple(row) for row in grid.map), i, playerTurn)] = (beta, 0)
            return (beta,0)

    # expectimax with alpha-beta pruning
    def expectimax(self, grid, movesLeft, playerTurn, alpha, beta, probOfReaching=1):
        #self.totalCalls += 1
        #check if this state has been computed to at least the required depth
        for i in range(movesLeft, 10):
            if (tuple(tuple(row) for row in grid.map), i, playerTurn) in self.memo:
                #self.MemoCalls += 1
                return self.memo[(tuple(tuple(row) for row in grid.map), i, playerTurn)]
        if playerTurn:
            #player moveset
            moveset = grid.getAvailableMoves([1,2,3,0])
        else:
            #computer moveset consists of placing 2 or 4 into an empty cell
            moveset = [(i,j) for i in [2,4] for j in grid.getAvailableCells()]
        #if not enough time left, cut off recursion
        if (time.process_time()-self.timer > self.time_limit):
            self.timed_out = True
            movesLeft = 0
        #if no more moves or at max depth or probability of reaching this node < 1/10000, cut off recursion
        #print(probOfReaching)
        if movesLeft == 0 or len(moveset) == 0 or probOfReaching < 0.0005:
            return (self.heuristic(grid), 0)
        #player turn
        if playerTurn:
            #default move is UP (0)
            maxMove = 0
            for move in moveset:
                #recursive call to evaluate child node
                newVal = self.expectimax(move[1], movesLeft-1, False, alpha, beta, probOfReaching)[0]
                #update max child value
                if newVal > alpha:
                    alpha = newVal
                    maxMove = move[0]
                #no pruning occurs in max nodes for expectimax variant of alpha-beta pruning
            #store computed result
            self.memo[(tuple(tuple(row) for row in grid.map), movesLeft, playerTurn)] = (alpha, maxMove)
            for i in range(movesLeft):
                if (tuple(tuple(row) for row in grid.map), i, playerTurn) in self.memo:
                    self.memo[(tuple(tuple(row) for row in grid.map), i, playerTurn)] = (alpha, maxMove)
            return (alpha, maxMove)
        #computer turn
        else:
            avgVal = 0
            #denominator for our weighted average
            totalWeight = sum([(0.9 if move[0]==2 else 0.1) for move in moveset])
            totalWeight = round(totalWeight,1)
            #cumulative denominator for our weighted average in case of pruning
            weightSoFar = 0
            for move in moveset:
                #max possible value the expectation node could take
                #assumes all unevaluated children have value UPPER_BOUND
                beta = (avgVal + (totalWeight-weightSoFar)*self.UPPER_BOUND) / totalWeight
                #alpha-beta pruning
                if alpha > beta:
                    break
                #place tile into cell to create child state
                newGrid = grid.clone()
                newGrid.setCellValue(move[1], move[0])
                #probability of getting a 2 is 90%, 4 is 10%
                prob = (0.9 if move[0]==2 else 0.1)
                #evaluate child node
                nodeValue = self.expectimax(newGrid, movesLeft-1, True, -float('inf'), beta, prob*probOfReaching/totalWeight)[0]
                #update avgVal and weightSoFar
                avgVal += prob*nodeValue
                weightSoFar += prob
            #to prevent division by 0
            if weightSoFar != 0:
                #divide weighted sum by total weight to get weighted average
                avgVal /= weightSoFar
            #store computed result
            self.memo[(tuple(tuple(row) for row in grid.map), movesLeft, playerTurn)] = (avgVal, 0)
            for i in range(movesLeft):
                if (tuple(tuple(row) for row in grid.map), i, playerTurn) in self.memo:
                    self.memo[(tuple(tuple(row) for row in grid.map), i, playerTurn)] = (avgVal, 0)
            return (avgVal,0)

#######################################
## Heuristics
#######################################
# Aryan
    #number of empty cells
    def h1(self, grid):
        return len(grid.getAvailableCells()) / 16

    def h2(self, grid):
        top5 = sorted([grid.map[i][j] for i in range(4) for j in range(4)], reverse=True)[:5]
        weights = [2,1.5,1,.5,.25]
        return sum(weights[i]*top5[i] for i in range(5)) / 10000

    #sum of tiles
    def sum_of_tiles(self, grid):
        return sum(sum(row) for row in grid.map) / 1600

    #push into corner [0][0]
    def h3(self, grid):
        gradient = [[ 5,  4,  3,  0],[ 1.5,  1,  1, 0],[ 0,  0, 0, 0],[ 0, 0, 0, 0]]
        val = 0
        for x in range(4):
            for y in range(4):
                val += gradient[x][y]*grid.map[x][y]
        return val / 2560

    #smoothness (sum of differences)
    #normalized by sum of tiles (using sum_of_tiles)
    def h4(self, grid):
        ans = 0
        for i in range(grid.size-1):
            for j in range(grid.size):
                ans -= abs(grid.getCellValue((i,j))-grid.getCellValue((i+1,j)))
                ans -= abs(grid.getCellValue((j,i))-grid.getCellValue((j,i+1)))
        return (ans / self.sum_of_tiles(grid)) / 4000


    def heuristic(self, grid):
        vals = [self.monotonicPatternHeuristic(grid), self.monotonicHeuristic(grid), self.clusterHeuristic(grid), self.mergeHeuristic(grid), self.openHeuristic(grid)]
        #print(vals, sum(vals))
        x = sum(vals[i]*self.weights[i] for i in range(len(vals)))
        '''
        if x > self.max_heur:
            self.max_heur = x
            print(self.max_heur)
            print(grid.map)
        '''
        if x > self.max_heur:
            self.max_heur = x

        return x
# Marin

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

        return score / (8192 * (4 ** 15))

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

        return score / (8192 * (4 ** 6))

    def mergeHeuristic(self,grid):
        """ Heuristics that rewards for the same values next to each other """
        score = 0
        for i in range(4):
            for j in range(4):
                curr = grid.getCellValue((i,j))
                neighborUp = grid.getCellValue((i - 1,j))
                neighborLeft = grid.getCellValue((i,j + 1))
                neighborDown = grid.getCellValue((i + 1,j))
                neighborRight = grid.getCellValue((i,j - 1))

                if curr == neighborUp:
                    score += curr * (4 ** 8)
                if curr == neighborLeft:
                    score += curr * (4 ** 8)
                if curr == neighborDown:
                    score += curr * (4 ** 8)
                if curr == neighborRight:
                    score += curr * (4 ** 8)

        return score / (64 * 2048 * (4 ** 8))

    def openHeuristic(self,grid):
        """ Heuristic that grants bonuses for the number of available tiles"""
        return len(grid.getAvailableCells()) / 16


    def clusterHeuristic(self,grid):
        """ Heuristic that penalizes tiles that have a big difference with their neightbors """

        penalty = 0
        for i in range(4):
            for j in range(4):
                # check for neighbors...
                neighborUp = grid.getCellValue((i - 1,j))
                neighborDown = grid.getCellValue((i + 1,j))
                neighborLeft = grid.getCellValue((i,j + 1))
                neighborRight = grid.getCellValue((i,j - 1))

                # Find absolute value of their differences
                if neighborUp != None:
                    penalty -= abs(grid.getCellValue((i,j)) - neighborUp)
                if neighborDown != None:
                    penalty -= abs(grid.getCellValue((i,j)) - neighborDown)
                if neighborLeft != None:
                    penalty -= abs(grid.getCellValue((i,j)) - neighborLeft)
                if neighborRight != None:
                    penalty -= abs(grid.getCellValue((i,j)) - neighborRight)

        # this will be assigned a negative because we are penalizing
        return penalty / (self.sum_of_tiles(grid) * 64000)

    def monotonicHeuristic(self, grid):
        """ Ensure tiles align monotonically """
        score = 0

        multiplier = 4

        for i in range(3):
            if multiplier == 0:
                break

            # these vars will be used for comparison between the values on lower edge
            curr = grid.getCellValue((3, 0 + i))
            prev = grid.getCellValue((3, 0 + i + 1))

            if curr >= prev:
                score += curr * multiplier
                multiplier *= 2
            else:
                multiplier = 0

        return score / (4096 * 4 + 2048 * 8 + 1024 * 16 + 512 * 32)

