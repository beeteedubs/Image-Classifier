# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import random
import math

PRINT = True


class MiraClassifier:
    """
  Mira classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 1.0
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys()  # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [1.02,1.04,1.08]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
    then store the weights that give the best accuracy on the validationData.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
        self.features = trainingData[0].keys()  # could be useful later
        trueWeight = {}
        bias = {}

        for y in self.legalLabels:
            trueWeight[y] = util.Counter()
            bias[y] = 0
            for value in self.features:
                trueWeight[y][value] = random.randrange(0, 2)
        self.weights = trueWeight
        self.bias = bias
        maxrange = len(trainingData)
        for iteration in range(self.max_iterations):
            incorrect = 0
            print "Starting iteration ", iteration, "..."
            randomrange = list(range(maxrange))
            random.shuffle(randomrange)
            for i in range(len(trainingData)):
                highestscore = None
                myY = None
                datum = trainingData[randomrange[i]]
                # datum = trainingData[i]
                scorearray = {}
                for y in self.legalLabels:
                    score = datum * self.weights[y] + bias[y]
                    if score > highestscore or highestscore is None:
                        highestscore = score
                        myY = y
                    perceptronScore = datum * trueWeight[y] + bias[y]
                    scorearray[y] = perceptronScore
                actualY = trainingLabels[randomrange[i]]
                # actualY = trainingLabels[i]
                # Wrong guess, update weights
                # actualY is the training label
                # myY is the y that i get  from list
                
                tau = min(Cgrid[0], ((self.weights[myY] - self.weights[actualY]) * datum + 1.0) / (2.0 * (datum * datum)))
                datumWithTau = datum
                for key in datum:
                  datumWithTau[key] = tau * datum[key]
                while myY != actualY:

                    bias[actualY] = bias[actualY] + 1
                    self.bias[actualY] = bias[actualY]
                    self.weights[actualY] = self.weights[actualY] + datumWithTau

                    bias[myY] = bias[myY] - 1
                    self.bias[myY] = bias[myY]
                    self.weights[myY] = self.weights[myY] - datumWithTau

                    score = (datum * self.weights[myY]) + bias[myY]
                    scorearray[myY] = score
                    score = (datum * self.weights[actualY]) + bias[actualY]
                    scorearray[actualY] = score
                    myY = max(scorearray, key=scorearray.get)
                    incorrect += 1
            print "number of incorrect: ", incorrect
            if incorrect == 0:
                break

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
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighOddsFeatures(self, label1, label2):
        """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        return featuresOdds
