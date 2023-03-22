# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
import sys

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    """
      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """
    """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.

    Some useful mdp methods you will use:
        mdp.getStates()
        mdp.getPossibleActions(state)
        mdp.getTransitionStatesAndProbs(state, action)
        mdp.getReward(state, action, nextState)
        mdp.isTerminal(state)
    """
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        self.values = util.Counter()
        for k in range(1,self.iterations + 1): # Begin at k = 1
            tempVals = util.Counter()
            for state in states:
                tempVals[state] = self.getMaxQValue(state) if not self.mdp.isTerminal(state) else 0
            self.values = tempVals

    def getMaxQValue(self, state):
        actionsAtState = self.mdp.getPossibleActions(state)
        maxQVal = -sys.maxsize
        for action in actionsAtState:
            maxQVal = max(maxQVal, self.computeQValueFromValues(state, action))
        return maxQVal
            


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        if(self.mdp.isTerminal(state)):
            return self.values[state]
        
        possibleStates = self.mdp.getTransitionStatesAndProbs(state,action)
        
        qVal = self.getActionNextStateSums(state, action, possibleStates)

        return qVal
    
    def getActionNextStateSums(self, state, action, possibleStates):
        if(len(possibleStates) == 0):
            return 0
        sumQVals = sum([self.getSample(state, action, nextState)*prob for nextState, prob in possibleStates])
        return sumQVals

    # returns r + γ*V_k(s')
    def getSample(self, state, action, nextState):
        sample = self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState]
        return sample
    
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # self.values -> {STATE, val}
        if(self.mdp.isTerminal(state)):
            return None
        actionsAtState = self.mdp.getPossibleActions(state)
        nextAction = None
        maxUtility = -sys.maxsize
        for action in actionsAtState:

            nextStates = self.mdp.getTransitionStatesAndProbs(state,action)
            q = self.getActionNextStateSums(state,action,nextStates)

            nextAction = action if q > maxUtility else nextAction
            maxUtility = max(q, maxUtility)

        return nextAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Write value iteration code here

        states = self.mdp.getStates()
        self.values = util.Counter()
        c = 0
        for k in range(1,self.iterations + 1): # Begin at k = 1
            if(c > len(states) - 1):
                c = 0

            state = states[c]
            self.values[state] = self.getMaxQValue(state) if not self.mdp.isTerminal(state) else 0
            c+=1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.

            Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessorMap = self.computeAllPredecessors()
        queue = util.PriorityQueue()

        for state in states:
            if(self.mdp.isTerminal(state)):
                continue
            maxQVal = self.getMaxQValue(state)
            diff = abs(self.values[state] - maxQVal)

            queue.push(state, -diff)
        
        for k in range(1,self.iterations + 1): # Begin at k = 1 
            if(queue.isEmpty()):
                break
            state = queue.pop()

            self.values[state] = self.getMaxQValue(state) 
            predecessors = predecessorMap[state]
            for predecessor in predecessors:
                if(self.mdp.isTerminal(predecessor)):
                    continue
                maxQVal = self.getMaxQValue(predecessor)
                diff = abs(self.values[predecessor] - maxQVal)
                if(diff > self.theta):
                    queue.update(predecessor, -diff)


    def computeAllPredecessors(self):
        states = self.mdp.getStates()
        predecessorMap = {}

        for state in states:
            if(self.mdp.isTerminal(state)):
                continue
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in nextStates:
                    if(nextState in predecessorMap):
                        predecessorMap[nextState].add(state)
                    else:
                        predecessorMap[nextState] = {state}

        return predecessorMap
