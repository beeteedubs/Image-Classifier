# perceptron.py
# -------------
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


# Perceptron implementation
import util
import random
import numpy as np
import statistics

PRINT = True


class PerceptronClassifier:
    """
  Perceptron classifier.
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;
    def classify(self, data):
        """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    Recall that a datum is a util.counter...
    """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum + self.bias[l]
            guesses.append(vectors.argMax())
        return guesses

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.
    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

        self.features = trainingData[0].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        trueWeight = {}
        bias = {}
        perceptronScore = 0

        for y in self.legalLabels:
          trueWeight[y] = util.Counter()
          bias[y] = 0
          for value in self.features:
            trueWeight[y][value] = random.randrange(0,1)
        self.weights = trueWeight
        self.bias = bias

        for iteration in range(self.max_iterations):
            incorrect = 0
            print "Starting iteration ", iteration, "..."
            randomrange = list(range(100))
            random.shuffle(randomrange)
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                bestScore = None
                bestY = None
                datum = trainingData[randomrange[i]]
                scorearray = {}
                for y in self.legalLabels:
                    score = datum * self.weights[y]
                    if score > bestScore or bestScore is None:
                        bestScore = score
                        bestY = y
                    # my way
                    perceptronScore = datum * trueWeight[y] + bias[y]
                    scorearray[y] = perceptronScore
                    # z = z + (dictionary.get((a,b)) * trueWeight[y].get((a,b)))
                actualY = trainingLabels[randomrange[i]]
                # Wrong guess, update weights
                while bestY != actualY:
                    bias[actualY] = bias[actualY] + 1
                    self.bias[actualY] = bias[actualY]
                    self.weights[actualY] = self.weights[actualY] + datum

                    bias[bestY] = bias[bestY] - 1
                    self.bias[bestY] = bias[bestY]
                    self.weights[bestY] = self.weights[bestY] - datum

                    score = (datum * self.weights[bestY]) + bias[bestY]
                    scorearray[bestY] = score
                    score = (datum * self.weights[actualY]) + bias[actualY]
                    scorearray[actualY] = score
                    bestY = max(scorearray, key=scorearray.get)
                    incorrect += 1
            print(incorrect)
            if incorrect == 0:
                break




        ##########MORE TESTING#############

        incorrect = 0

        dictionaryForLabels = []

        for y in self.legalLabels:
          line = []
          for i in range(len(trainingLabels)):
            if trainingLabels[i] == y:
              line.append(i)
          dictionaryForLabels.append(line)
        x = 0
        j = 0
        i = 0
        while j < len(dictionaryForLabels):
            while i < len(dictionaryForLabels[j]):
                datum = trainingData[dictionaryForLabels[j][i]]
                scorearray = {}
                for y in self.legalLabels:
                    perceptronScore = datum * trueWeight[y] + bias[y]
                    scorearray[y] = perceptronScore
                actualY = trainingLabels[dictionaryForLabels[j][i]]
                bestY = max(scorearray, key=scorearray.get)

                standardDeviation = statistics.stdev(scorearray[k] for k in scorearray)
                standardAverage = statistics.mean(scorearray[k] for k in scorearray)
                # Wrong guess, update weights
                while bestY != actualY:
                    self.bias[actualY] = self.bias[actualY] + 1
                    self.bias[actualY] = bias[actualY]
                    self.weights[actualY] = self.weights[actualY] + datum

                    self.bias[bestY] = self.bias[bestY] - 1
                    self.weights[bestY] = self.weights[bestY] - datum

                    score = (datum * self.weights[bestY]) + bias[bestY]
                    scorearray[bestY] = score
                    score = (datum * self.weights[actualY]) + bias[actualY]
                    scorearray[actualY] = score
                    bestY = max(scorearray, key=scorearray.get)
                    incorrect += 1
                    print("changed")
                y = 0
                while y < 10:
                    # Basically: self.weights[y] = self.weights[y] - datum(.5)
                    if scorearray.get(y) > (standardAverage + standardDeviation) and y != actualY:
                        for k, v in self.weights[y].items():
                            self.weights[y][k] = v - datum[k] * .25
                        score = (datum * self.weights[y]) + bias[y]
                        scorearray[y] = score
                        incorrect += 1
                    score = scorearray.get(max(scorearray, key=scorearray.get))
                    if (scorearray.get(y) > (score - standardDeviation)) and y != actualY:
                        # Basically: self.weights[y] = self.weights[y] - datum(.25)
                        for k, v in self.weights[y].items():
                            self.weights[y][k] = v - datum[k] * .25
                        # Basically: self.weights[y] = self.weights[y] +  datum(.25)
                        for k, v in self.weights[actualY].items():
                            self.weights[actualY][k] = v + datum[k] * .25
                        score = (datum * self.weights[y]) + bias[y]
                        scorearray[y] = score
                        score = (datum * self.weights[actualY]) + bias[actualY]
                        scorearray[actualY] = score
                        incorrect += 1
                    if scorearray.get(y) > 0 and y != actualY:
                        # Basically: self.weights[y] = self.weights[y] - datum(.15)
                        # Reduce weights by 10%
                        for k, v in self.weights[y].items():
                            self.weights[y][k] = v - datum[k] * .1
                        score = (datum * self.weights[y]) + bias[y]
                        scorearray[y] = score
                        incorrect += 1
                        print("reduced", y)
                        y = -1
                    if scorearray.get(actualY) < 0:
                        # Increase weights by 50%
                        for k, v in self.weights[actualY].items():
                            self.weights[actualY][k] = v + datum[k] * .5
                        score = (datum * self.weights[actualY]) + bias[actualY]
                        scorearray[actualY] = score
                        incorrect += 1
                        print("increased", actualY)
                        y = -1
                    y += 1
                i += 1
            i = 0
            if incorrect != 0:
                incorrect = 0
                j = 0
                i = 0
            else:
                j += 1
        print("donzo")
        ####Final Iteration###
        for iteration in range(self.max_iterations):
            incorrect = 0
            print "Starting iteration ", iteration, "..."
            randomrange = list(range(100))
            random.shuffle(randomrange)
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                bestScore = None
                bestY = None
                datum = trainingData[randomrange[i]]
                scorearray = {}
                for y in self.legalLabels:
                    score = datum * self.weights[y]
                    if score > bestScore or bestScore is None:
                        bestScore = score
                        bestY = y
                    # my way
                    perceptronScore = datum * trueWeight[y] + bias[y]
                    scorearray[y] = perceptronScore
                    # z = z + (dictionary.get((a,b)) * trueWeight[y].get((a,b)))
                actualY = trainingLabels[randomrange[i]]
                # Wrong guess, update weights
                while bestY != actualY:
                    bias[actualY] = bias[actualY] + 1
                    self.bias[actualY] = bias[actualY]
                    self.weights[actualY] = self.weights[actualY] + datum

                    bias[bestY] = bias[bestY] - 1
                    self.bias[bestY] = bias[bestY]
                    self.weights[bestY] = self.weights[bestY] - datum

                    score = (datum * self.weights[bestY]) + bias[bestY]
                    scorearray[bestY] = score
                    score = (datum * self.weights[actualY]) + bias[actualY]
                    scorearray[actualY] = score
                    bestY = max(scorearray, key=scorearray.get)
                    incorrect += 1
            print(incorrect)
            if incorrect == 0:
                break
        #guesses = perceptron.classify(validationData)
        #correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        #print(correct)


    def findHighWeightFeatures(self, label):
        """
    Returns a list of the 100 features with the greatest weight for some label
    """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        #featuresWeights = self.weights[label].sortedKeys()[0z:100]
        featuresWeights = [k for k, v in sorted(self.weights[label].items(), key=lambda (k, v): (-v, k))][0:100]

        return featuresWeights