import csv
import pickle
import random
from textblob.classifiers import NaiveBayesClassifier


class NaiveBayesTrainer:
    tweetSentiments = []
    trainingData = []
    testData = []

    def split_data(self, lower_bound, upper_bound):
        self.trainingData = self.tweetSentiments[lower_bound:upper_bound]

    def parse(self):
        path = '/Users/Max/Desktop/Machine Learning Sentiment/Sentiment Analysis Dataset.csv';
        with open(path, 'r') as file:
            count = 0
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if count > 0:
                    if row[1] == '0':
                        self.tweetSentiments.append((row[3], 'neg'))
                    else:
                        self.tweetSentiments.append((row[3], 'pos'))
                count += 1

    def train_model(self):
        bayes_classifier = NaiveBayesClassifier(self.trainingData)
        file = open('naive-bayes-classifier.pickle', 'wb')
        pickle.dump(bayes_classifier, file)
        file.close()
        return classifier


if __name__ == '__main__':
    # Bayes Classifier
    try:
        f = open('naive-bayes-classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
        dist = classifier.prob_classify('School is amazing')
        print(dist.prob('pos'))
        print(dist.prob('neg'))
    except(OSError, IOError, FileNotFoundError) as error:
        bayesTrainer = NaiveBayesTrainer()
        bayesTrainer.parse()
        bayesTrainer.split_data(0, 10000)
        classifier = bayesTrainer.train_model()
        dist = classifier.prob_classify('School is really bad')
        print(dist.prob('pos'))
        print(dist.prob('neg'))

    # Deep Learning Classifier