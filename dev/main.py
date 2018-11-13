import csv
import pickle
import pandas
import random

from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob.classifiers import NaiveBayesClassifier


class NaiveBayesTrainer:
    tweetSentiments, tweetTexts = [], []
    trainingData = pandas.DataFrame()

    def parse(self):
        path = '/Users/Max/Desktop/Machine Learning Sentiment/Sentiment Analysis Dataset.csv';
        with open(path, 'r') as file:
            count = 0
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if count > 0:
                    if row[1] == '0':
                        self.tweetSentiments.append('neg')
                    else:
                        self.tweetSentiments.append('pos')
                    self.tweetTexts.append(row[3])
                count += 1

        self.tweetTexts = self.tweetTexts[0:1000]
        self.tweetSentiments = self.tweetSentiments[0:1000]
        self.trainingData['text'] = self.tweetTexts
        self.trainingData['label'] = self.tweetSentiments

        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(self.trainingData['text'], self.trainingData['label'])

        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)

        return train_x, valid_x, train_y, valid_y


if __name__ == '__main__':
    # Bayes Classifier
        bayesTrainer = NaiveBayesTrainer()
        train_x, valid_x, train_y, valid_y = bayesTrainer.parse()
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(bayesTrainer.trainingData['text'])
        xtrain_count = count_vect.transform(train_x)
        xvalid_count = count_vect.transform(valid_x)
        multinomialNB = naive_bayes.MultinomialNB()
        multinomialNB.fit(xtrain_count, train_y)
        predictions = multinomialNB.predict(xvalid_count)
        print(train_x)
        print(predictions)
        print(metrics.accuracy_score(predictions, valid_y))

    # Deep Learning Classifier