# Marin P. Marinov Homework 2

## Implementation details

### Algorithm
- After conducting several tests, I realized using expectimax which is a special case of expectiminimax where we take max and average only was more efficient
  - This makes sense as the AI plays randomly and there is technically only one player, it better fits the game
- The function used is called **expectiAlphaBeta** which performs alpha beta pruning on expectimax

### Heuristics
- I used 5 heuristic and did a weighted sum of them, they are:
  - Monotomic heuristic (makes sure values increase/decrease monotomically)
  - pattern heuristic (Highest number is in a corner)
  - cluster heuristic (Penalizes if tiles with a large difference in their values are together)
  - open heuristic (Gives reward for open tiles)
  - merge heuristic (gives reward for positions where there are tiles of the same value)
- I weighted the penalty the most to try to ensure that the highest squares stay together 
- The above heuristics are inspired by the information of the below posts and papers
  - https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
  - https://home.cse.ust.hk/~yqsong/teaching/comp3211/projects/2017Fall/G11.pdf
  - http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

### Findings
- Highest score I got while testing: 8192
- Greatest number of 2048 in a row: 7
- Expectimax with a lower depth performed better than minimax with alpha beta pruning with higher depth
- Expectimax with alpha beta pruning seems to be most efficient 

# Tasks Completed 
- Expectiminimax
- minimax and expectiminimax with Alph beta pruning
- Heuristic function
- Heuristic functions sum with weights
