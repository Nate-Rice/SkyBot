import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer  # used to stem words
from flask import Flask, render_template, request
from stop_callback import EarlyStoppingCallback
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
app = Flask(__name__)


class Chatter:
    def __init__(self):
        self.model = []
        self.words = []
        self.labels = []
        self.data = None
        self.stemmer = LancasterStemmer()
        self.docs_pattern = []
        self.docs_intents = []

    def model_load(self):
        with open("intents.json") as file:
            self.data = json.load(file)

        try:
            with open("models/data.pickle", "rb") as f:
                self.words, self.labels, training, output = pickle.load(f)
        except:
            for intent in self.data["intents"]:  # Preprocessing data
                for pattern in intent["patterns"]:
                    words_pattern = nltk.word_tokenize(pattern)  # Getting all the words and placing them into words list
                    self.words.extend(words_pattern)
                    self.docs_pattern.append(words_pattern)
                    self.docs_intents.append(intent["tag"])

                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

            self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != "?"]  # Getting the stem of the word
            self.words = sorted(list(set(self.words)))  # Removing duplicate words and converting back to list

            self.labels = sorted(self.labels)

            # One hot encoded
            training = []
            output = []

            output_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(self.docs_pattern):
                bag = []
                pattern_words = [self.stemmer.stem(w) for w in doc]

                for w in self.words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                output_row = list(output_empty)
                output_row[self.labels.index(self.docs_intents[x])] = 1

                training.append(bag)
                output.append(output_row)

            training = numpy.array(training)  # Change lists into arrays so that we can feed into model
            output = numpy.array(output)

            with open("models/data.pickle", "wb") as f:
                pickle.dump((self.words, self.labels, training, output), f)

        tensorflow.reset_default_graph()

        n_network = tflearn.input_data(shape=[None, len(training[0])])  # Input data
        n_network = tflearn.fully_connected(n_network, 32)  # 32 neurons for hidden layer
        n_network = tflearn.fully_connected(n_network, 32)
        n_network = tflearn.fully_connected(n_network, len(output[0]), activation="softmax")  # Output layer.
        n_network = tflearn.regression(n_network)

        self.model = tflearn.DNN(n_network)

        try:
            self.model.load("models/model.tflearn")
        except:
            self.model = tflearn.DNN(n_network)
            early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.998, val_epoch_thresh=500)
            try:
                self.model.fit(training, output, n_epoch=2000, batch_size=4, show_metric=True, snapshot_epoch=False,
                               callbacks=early_stopping_cb)
            except StopIteration:
                print("Caught callback exception. Returning control to user program.")
            self.model.save("models/model.tflearn")

    def stem_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence, words):
        bag = [0 for _ in range(len(words))]

        sentence_words = self.stem_sentence(sentence)

        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1

        return numpy.array(bag)

    def chat(self, user_text):
        probability_results = self.model.predict([self.bag_of_words(user_text, self.words)])[0]
        results_index = numpy.argmax(probability_results)
        tag = self.labels[results_index]

        if probability_results[results_index] > 0.5:
            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            return random.choice(responses)
        else:
            return "Sorry, I don't understand. Please try a different input."


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def chatty():
    c = Chatter()
    c.model_load()
    while True:
        user_text = request.args.get('msg')
        return c.chat(user_text)

#def main():
#    c = Chatter()
#    c.model_load()


if __name__ == "__main__":
    #main()
    app.run()

