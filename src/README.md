# 2048-AI final project

## To Run
`python tester.py`

## Implementation details

### Algorithm
Iterative-deepening expectimax

### Heuristics
- I used 4 heuristic and did a weighted sum of them, they are:
  - pattern heuristic (Highest number is in a corner)
  - cluster heuristic (Penalizes if tiles with a large difference in their values are together)
  - open heuristic (Gives reward for open tiles)
  - merge heuristic (gives reward for positions where there are tiles of the same value)
- Weights are assigned based on CMA-ES
- The above heuristics are inspired by the information of the below posts and papers
  - https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
  - https://home.cse.ust.hk/~yqsong/teaching/comp3211/projects/2017Fall/G11.pdf
  - http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

### Findings
- Highest score we got while testing: 8192 with a 2048 tile
- Greatest number of 2048 in a row: 8

