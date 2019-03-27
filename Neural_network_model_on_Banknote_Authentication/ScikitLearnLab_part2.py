import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

class NeuralNet:
    def __init__(self, url):
        names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        self.df = pd.read_csv(url, header=None, names=names)

    def plotGraph(self):
        self.df.iloc[:, 0:4].hist()
        correlations = self.df.iloc[:, 0:4].corr()
        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, 4, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.df.iloc[:, 0:4].columns)
        ax.set_yticklabels(self.df.iloc[:, 0:4].columns)
        plt.title("Correlation Matrix \n")
        plt.show()
        #create a scatterplot
        scatter_matrix(self.df.iloc[:, 0:4])
        plt.show()

    def preProcess(self):
        nrows, ncols = self.df.shape[0], self.df.shape[1]
        X = self.df.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        Y = self.df.iloc[:, ncols - 1].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=np.random)
        return X_train, X_test, Y_train, Y_test

    # Below is the training function
    def train(self, X_train, Y_train):
        model = MLPClassifier(hidden_layer_sizes=(7, 7), alpha=0.0003, max_iter=500)
        model.fit(X_train, Y_train)
        return model

    # predict on test dataset
    def predict(self, model, X_test, Y_test):
        predictions = model.predict(X_test)
        print("Intercepts for neurons :", model.intercepts_)
        print("Weights for neurons :", model.coefs_)
        print("Confusion Matrix :")
        print(confusion_matrix(Y_test, predictions))
        print("Classification Report :")
        print(classification_report(Y_test, predictions))

if __name__ == "__main__":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    neural_network = NeuralNet(url)
    neural_network.plotGraph()
    X_train, X_test, Y_train, Y_test = neural_network.preProcess();
    model = neural_network.train(X_train, Y_train)
    neural_network.predict(model, X_test, Y_test)