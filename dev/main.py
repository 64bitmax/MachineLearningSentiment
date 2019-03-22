import configparser
import re
import random
import pymongo
import pickle
import pandas as pd
import json

from sklearn import model_selection, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MongoConnector:
    def __init__(self, uri):
        self.mongoClient = pymongo.MongoClient(uri)

    def findDatabases(self):
        return self.mongoClient.list_database_names()

    def findCollections(self, database_name):
        return self.mongoClient[database_name].collection_names()

    def retrieveData(self, database_name, table_name):
        db = self.mongoClient[database_name]
        table = db[table_name]

        storage = []
        for item in table.find():
            storage.append(item)

        return storage

    def sendData(self, database_name, table_name, data):
        db = self.mongoClient[database_name]
        table = db[table_name]
        table.insert_many(data)


class MachineLearning:
    def preprocess(self, text):
        replace_no_space = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
        replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        text = [replace_no_space.sub("", line.lower()) for line in text]
        text = [replace_with_space.sub(" ", line) for line in text]

        return text

    def train(self):
        # ----- Import all training data and validation data -----
        tweet_sentiments, tweet_texts = [], []
        tweet_csv = pd.read_csv('Sentiment Analysis Dataset.csv', delimiter=',',
                                names=['ItemID', 'Sentiment', 'SentimentSource', 'SentimentText'], skiprows=1)

        tweet_sentiments = tweet_csv['Sentiment']
        tweet_texts = tweet_csv['SentimentText']

        train_movie_texts_list = []
        train_movie_sentiments_list = [1 if i < 12500 else 0 for i in range(25000)]
        for line in open('/Users/Max/Desktop/MachineLearningSentiment/aclImdb/movie_data/full_train.txt', 'r'):
            train_movie_texts_list.append(line)

        valid_movie_texts_list = []
        valid_movie_sentiments_list = [1 if i < 12500 else 0 for i in range(25000)]
        for line in open('/Users/Max/Desktop/MachineLearningSentiment/aclImdb/movie_data/full_test.txt', 'r'):
            valid_movie_texts_list.append(line.strip())

        emojis_dict = {}
        emojis = []
        emojisentiments = []

        with open('emojis.json', 'r') as file:
            emojis_dict = json.load(file)

        for item in emojis_dict:
            emojis.append(item['emoji'])
            if item['polarity'] > 0:
                emojisentiments.append(1)
            elif item['polarity'] < 0:
                emojisentiments.append(0)

        # ----- Perform basic pre-processing for optimization -----
        tweet_texts = self.preprocess(tweet_texts)
        train_movie_texts_list = self.preprocess(train_movie_texts_list)
        valid_movie_texts_list = self.preprocess(valid_movie_texts_list)

        # ----- Create the training and test sets -----
        train_movie_texts = pd.Series(train_movie_texts_list)
        train_movie_sentiments = pd.Series(train_movie_sentiments_list)

        valid_movie_texts = pd.Series(valid_movie_texts_list)
        valid_movie_sentiments = pd.Series(valid_movie_sentiments_list)

        train_tweet_text_list, valid_tweet_text_list, train_tweet_sentiments_list, valid_tweet_sentiments_list = model_selection.train_test_split(
            tweet_texts, tweet_sentiments)

        train_tweet_text = pd.Series(train_tweet_text_list)
        valid_tweet_text = pd.Series(valid_tweet_text_list)
        train_tweet_sentiments = pd.Series(train_tweet_sentiments_list)
        valid_tweet_sentiments = pd.Series(valid_tweet_sentiments_list)

        train_emoji = pd.Series(emojis)
        train_emoji_sentiments = pd.Series(emojisentiments)

        # ----- Concatenate training data -----
        train_x_init = train_tweet_text.append([train_movie_texts])
        train_y_init = train_tweet_sentiments.append([train_movie_sentiments])
        train_x = train_x_init.append([train_emoji])
        train_y = train_y_init.append([train_emoji_sentiments])
        valid_x = valid_tweet_text.append([valid_movie_texts])
        valid_y = valid_tweet_sentiments.append([valid_movie_sentiments])

        # ----- Shuffle training data -----
        training = list(zip(train_x, train_y))
        validation = list(zip(valid_x, valid_y))
        random.shuffle(training)
        random.shuffle(validation)
        train_x, train_y = zip(*training)
        valid_x, valid_y = zip(*validation)

        # ----- Vectorize the Training Data -----
        count_vect = TfidfVectorizer()
        count_vect.fit_transform(train_x)

        # ----- Transformation to a Document-Term Matrix -----
        xtrain_doc = count_vect.transform(train_x)
        xvalid_doc = count_vect.transform(valid_x)

        # ----- Multinomial Naive Bayes Classifier -----
        multinomialNB = naive_bayes.MultinomialNB()
        multinomialNB.fit(xtrain_doc, train_y)

        # ----- Predict on Validation Data -----
        bayesPredictions = multinomialNB.predict(xvalid_doc)

        # ----- Compute Accuracy of Validation Tests -----
        bayesAccuracy = metrics.accuracy_score(bayesPredictions, valid_y)

        # ----- Linear Regression Classifier -----
        logRegression = LogisticRegression(C=0.05)
        logRegression.fit(xtrain_doc, train_y)

        # ----- Predict on Validation Data -----
        logRegressionPredictions = logRegression.predict(xvalid_doc)
        logRegressionAccuracy = metrics.accuracy_score(logRegressionPredictions, valid_y)

        regression = open("regression.pickle", 'wb')
        pickle.dump(logRegression, regression)
        regression.close()

        bayes = open("bayes.pickle", 'wb')
        pickle.dump(multinomialNB, bayes)
        bayes.close()

        vect = open("vect.pickle", 'wb')
        pickle.dump(count_vect, vect)
        vect.close()

        return bayesPredictions, logRegressionPredictions

    def predict_years(self):
        # ----- Load trained models -----
        file = open("bayes.pickle", 'rb')
        multinomialNB = pickle.load(file)
        file.close()

        file = open("regression.pickle", 'rb')
        logRegression = pickle.load(file)
        file.close()

        file = open("vect.pickle", 'rb')
        count_vect = pickle.load(file)
        file.close()

        config = configparser.ConfigParser()
        config.read('config.ini')
        uri = config['CLOUD']['uri']

        academic_support = config['CLOUD']['academic-support-table']
        assessment_and_feedback = config['CLOUD']['assessment-and-feedback-table']
        learning_community = config['CLOUD']['learning-community-table']
        learning_opportunities = config['CLOUD']['learning-opportunities-table']
        learning_resources = config['CLOUD']['learning-resources-table']
        organisation_and_management = config['CLOUD']['organisation-and-management-table']
        overall_teaching = config['CLOUD']['overall-teaching-table']
        student_voice = config['CLOUD']['student-voice-table']
        overall = config['CLOUD']['overall-table']

        # ----- Use data from database -----
        mongo = MongoConnector(uri)
        dbs = mongo.findDatabases()

        for db_name in dbs:
            if db_name.__contains__('_tweets'):
                academic_support_tweets = mongo.retrieveData(db_name, academic_support)
                assessment_and_feedback_tweets = mongo.retrieveData(db_name, assessment_and_feedback)
                learning_community_tweets = mongo.retrieveData(db_name, learning_community)
                learning_opportunities_tweets = mongo.retrieveData(db_name, learning_opportunities)
                learning_resources_tweets = mongo.retrieveData(db_name, learning_resources)
                organisation_and_management_tweets = mongo.retrieveData(db_name, organisation_and_management)
                overall_teaching_tweets = mongo.retrieveData(db_name, overall_teaching)
                student_voice_tweets = mongo.retrieveData(db_name, student_voice)
                overall_tweets = mongo.retrieveData(db_name, overall)

                tweets_list = [academic_support_tweets, assessment_and_feedback_tweets, learning_community_tweets,
                               learning_opportunities_tweets, learning_resources_tweets, organisation_and_management_tweets,
                                overall_teaching_tweets, student_voice_tweets, overall_tweets]

                counter = 0
                for tweet_list in tweets_list:
                    if len(tweet_list) > 0:
                        tweet_texts = []
                        for tweet in tweet_list:
                            if tweet["tweet"] != "":
                                tweet_texts.append(tweet["tweet"])

                        tweets = self.preprocess(tweet_texts)
                        tweets = count_vect.transform(tweets)

                        regressionResults = logRegression.predict(tweets).tolist()
                        bayesResults = multinomialNB.predict(tweets).tolist()

                        regression_final = []
                        bayes_final = []

                        for i in range(len(regressionResults)):
                            list_category = ""
                            if counter == 0:
                                list_category = academic_support
                            elif counter == 1:
                                list_category = assessment_and_feedback
                            elif counter == 2:
                                list_category = learning_community
                            elif counter == 3:
                                list_category = learning_opportunities
                            elif counter == 4:
                                list_category = learning_resources
                            elif counter == 5:
                                list_category = organisation_and_management
                            elif counter == 6:
                                list_category = overall_teaching
                            elif counter == 7:
                                list_category = student_voice
                            elif counter == 8:
                                list_category = overall

                            reg = {"category": list_category, "tweet": tweet_texts[i],
                                   "sentiment": regressionResults[i]}
                            bayes = {"category": list_category, "tweet": tweet_texts[i], "sentiment": bayesResults[i]}
                            regression_final.append(reg)
                            bayes_final.append(bayes)

                        mongo.sendData(db_name + "_predictions", "tweet_sentiments_regression", regression_final)
                        mongo.sendData(db_name + "_predictions", "tweet_sentiments_bayes", bayes_final)
                        counter = counter + 1
                    else:
                        counter = counter + 1
                        continue

    def predict_current(self):
        # ----- Load trained models -----
        file = open("bayes.pickle", 'rb')
        multinomialNB = pickle.load(file)
        file.close()

        file = open("regression.pickle", 'rb')
        logRegression = pickle.load(file)
        file.close()

        file = open("vect.pickle", 'rb')
        count_vect = pickle.load(file)
        file.close()

        config = configparser.ConfigParser()
        config.read('config.ini')
        uri = config['CLOUD']['uri']
        db = config['CLOUD']['db']

        academic_support = config['CLOUD']['academic-support-table']
        assessment_and_feedback = config['CLOUD']['assessment-and-feedback-table']
        learning_community = config['CLOUD']['learning-community-table']
        learning_opportunities = config['CLOUD']['learning-opportunities-table']
        learning_resources = config['CLOUD']['learning-resources-table']
        organisation_and_management = config['CLOUD']['organisation-and-management-table']
        overall_teaching = config['CLOUD']['overall-teaching-table']
        student_voice = config['CLOUD']['student-voice-table']
        overall = config['CLOUD']['overall-table']

        # ----- Use data from database -----
        mongo = MongoConnector(uri)
        academic_support_tweets = mongo.retrieveData(db, academic_support)
        assessment_and_feedback_tweets = mongo.retrieveData(db, assessment_and_feedback)
        learning_community_tweets = mongo.retrieveData(db, learning_community)
        learning_opportunities_tweets = mongo.retrieveData(db, learning_opportunities)
        learning_resources_tweets = mongo.retrieveData(db, learning_resources)
        organisation_and_management_tweets = mongo.retrieveData(db, organisation_and_management)
        overall_teaching_tweets = mongo.retrieveData(db, overall_teaching)
        student_voice_tweets = mongo.retrieveData(db, student_voice)
        overall_tweets = mongo.retrieveData(db, overall)

        tweets_list = [academic_support_tweets, assessment_and_feedback_tweets, learning_community_tweets,
                       learning_opportunities_tweets, organisation_and_management_tweets, learning_resources_tweets,
                       overall_teaching_tweets, student_voice_tweets, overall_tweets]

        counter = 0
        for tweet_list in tweets_list:
            tweet_texts = []
            for tweet in tweet_list:
                if tweet["tweet"] != "":
                    tweet_texts.append(tweet["tweet"])

            tweets = self.preprocess(tweet_texts)
            tweets = count_vect.transform(tweets)

            regressionResults = logRegression.predict(tweets).tolist()
            bayesResults = multinomialNB.predict(tweets).tolist()

            regression_final = []
            bayes_final = []

            for i in range(len(regressionResults)):
                list_category = ""
                if counter == 0:
                    list_category = academic_support
                elif counter == 1:
                    list_category = assessment_and_feedback
                elif counter == 2:
                    list_category = learning_community
                elif counter == 3:
                    list_category = learning_opportunities
                elif counter == 4:
                    list_category = learning_resources
                elif counter == 5:
                    list_category = organisation_and_management
                elif counter == 6:
                    list_category = overall_teaching
                elif counter == 7:
                    list_category = student_voice
                elif counter == 8:
                    list_category = overall

                reg = {"category": list_category, "tweet": tweet_texts[i], "sentiment": regressionResults[i]}
                bayes = {"category": list_category, "tweet": tweet_texts[i], "sentiment": bayesResults[i]}
                regression_final.append(reg)
                bayes_final.append(bayes)

            mongo.sendData("predictions", "tweet_sentiments_regression", regression_final)
            mongo.sendData("predictions", "tweet_sentiments_bayes", bayes_final)
            counter = counter + 1


if __name__ == "__main__":
    ml = MachineLearning()
    cmd = input("Train a new model or load existing [TRAIN, PREDICT]: ")
    if cmd == "TRAIN":
        ml.train()
    elif cmd == "PREDICT_CURRENT":
        ml.predict_current()
    elif cmd == "PREDICT_YEARS":
        ml.predict_years()