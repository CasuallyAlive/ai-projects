# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        min_distance = getClosestFood(newPos,newFood.asList())
        
        newGhostStates = successorGameState.getGhostStates()

        ghostDistances = 0
        for ghost in newGhostStates:
            distanceFromPacman = manhattanDistance(newPos, ghost.getPosition())
            ghostPos = ghost.getPosition()
            if distanceFromPacman <= 3 and ghost.scaredTimer == 0:
                ghostDistances = -sys.maxsize-1
                break
            ghostDistances += distanceFromPacman+ghost.scaredTimer
              

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore() + ghostDistances - min_distance

# Gets position of food closest to pacman
def getClosestFood(pacmanPos, foodPositions):
    min_d = sys.maxsize
    for foodPos in foodPositions:
        temp_d = manhattanDistance(pacmanPos, foodPos)
        if temp_d < min_d:
            min_d = temp_d
    if(min_d == sys.maxsize):
        return 0
    return min_d

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
 
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    # Recursive minimax function
    
    def miniMax(self, state, depth, agent):
        if state.isLose() or state.isWin() or depth == self.depth:
            return (self.evaluationFunction(state), None)
        if agent == 0:
            maxVal = -sys.maxsize
            maxAction = None
            for action in state.getLegalActions(agent):
                newScore, _ = self.miniMax(state.generateSuccessor(agent, action), depth, 1)
                maxAction = action if maxVal <= newScore else maxAction
                maxVal = max(maxVal, newScore)
            return (maxVal, maxAction)
        else:
            nextAgent = agent + 1 if agent <= state.getNumAgents()-2 else 0
            depth = depth + 1 if nextAgent == 0 else depth
            minVal = sys.maxsize
            minAction = None
            for action in state.getLegalActions(agent):
                newScore, _ = self.miniMax(state.generateSuccessor(agent, action), depth, nextAgent)     
                minAction = action if minVal >= newScore else minAction
                minVal = min(minVal, newScore)
            return (minVal, minAction) 
        

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        score, action = self.miniMax(gameState, 0, self.index)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action = self.fastMiniMax(gameState, 0, -sys.maxsize, sys.maxsize, self.index)
        return action
    
    def fastMiniMax(self, state, depth, alpha, beta, agent):
        isMax = agent == 0
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluationFunction(state), None
        if isMax:
            val, maxAction, alpha = self.max_v(state, depth, agent, alpha, beta)
            return val, maxAction
        else:
            val, minAction, beta = self.min_v(state, depth, agent, alpha, beta)
            return val, minAction
        
    def max_v(self, state, depth, agent, alpha, beta):
        maxVal = -sys.maxsize
        maxAction = None
        for action in state.getLegalActions(agent):
            newScore, _ = self.fastMiniMax(state.generateSuccessor(agent, action), depth, alpha, beta, 1)
            maxAction = action if maxVal <= newScore else maxAction
            maxVal = max(maxVal, newScore)
            if(maxVal > beta):
                return maxVal, maxAction, alpha
            alpha = max(alpha, maxVal)
        return maxVal, maxAction, alpha

    def min_v(self, state, depth, agent, alpha, beta):
        nextAgent = agent + 1 if agent <= state.getNumAgents()-2 else 0
        depth = depth + 1 if nextAgent == 0 else depth
        minVal = sys.maxsize
        minAction = None
        for action in state.getLegalActions(agent):
            newScore, _ = self.fastMiniMax(state.generateSuccessor(agent, action), depth, alpha, beta, nextAgent)     
            minAction = action if minVal >= newScore else minAction
            minVal = min(minVal, newScore)
            if(minVal < alpha):
                return minVal, minAction, beta
            beta = min(beta, minVal)
        return minVal, minAction, beta

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agent = self.index
        if agent != 0:
            return random.choice(gameState.getLegalActions(agent))

        return self.expectimax(gameState, agent, 0)[1]
    
    def expectimax(self, state, agent, depth):
        isMax = agent == 0

        if(depth == self.depth or state.isWin() or state.isLose()):
            return self.evaluationFunction(state), None
        
        if(isMax):
            return self.max_v(state, agent, depth)
        
        return self.chance_v(state, agent, depth)
        
        
    
    def chance_v(self, state, agent, depth):
        nextAgent = agent + 1 if agent <= state.getNumAgents()-2 else 0
        nextDepth = depth + 1 if nextAgent == 0 else depth
        legalMoves = state.getLegalActions(agent)

        prob = 1.0 / float(len(legalMoves))
        expectedValue = float()
        for action in legalMoves:
            nextState = state.generateSuccessor(agent, action)
            val, _ = self.expectimax(nextState, nextAgent, nextDepth)
            expectedValue += prob*val

        return expectedValue, None

    def max_v(self, state, agent, depth):

        legalMoves = state.getLegalActions(agent)
        results = []
        for action in legalMoves:
            nextState = state.generateSuccessor(agent, action)
            expectedVal, _ = self.expectimax(nextState, 1, depth)
            results.append((expectedVal, action))

        return max(results, key=lambda t: t[0])
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    ghostVal = min([getGhostVal(ghost, pacmanPos) for ghost in ghostStates])

    return currentGameState.getScore() + ghostVal - getClosestFood(pacmanPos, food.asList())



def getGhostVal(ghost, pacPos):
    pgDistance = manhattanDistance(ghost.getPosition(), pacPos)
    if(pgDistance <= 1):
        v = sys.maxsize if ghost.scaredTimer >= 1 else -sys.maxsize
        return v
    return 0

# Abbreviation
better = betterEvaluationFunction
