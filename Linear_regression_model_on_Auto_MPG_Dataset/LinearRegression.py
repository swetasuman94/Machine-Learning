
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class LinearRegression:
    def __init__(self, url):
        names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
        self.df = pd.read_csv(url, header=None, names=names)

    # Below is the pre-process function
    def preProcess(self):
        self.df = self.df.dropna()
        self.df.drop('car name',axis=1,inplace=True)
        self.nrows, self.ncols = self.df.shape[0], self.df.shape[1]
        X = self.df.iloc[:, 1:(self.ncols)].values.reshape(self.nrows, self.ncols - 1)
        Y = self.df.iloc[:, 0].values.reshape(self.nrows, 1)
        X_scaled = preprocessing.scale(X)
        Y_scaled = preprocessing.scale(Y)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.35, random_state=np.random)
        return x_train, x_test, y_train, y_test

    # Below is the training function
    def train(self, x_train, y_train):
        model = linear_model.LinearRegression()
        print(y_train)
        model.fit(x_train, y_train)
        coefficients = model.coef_
        intercept = model.intercept_[0]
        print("Estimated intercept :", intercept)
        print("Length of Estimated Coefficient :", len(coefficients[0]))
        names = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
        df_coeff = pd.DataFrame(zip(list(names), list(coefficients[0])), columns=['Features', 'Coefficients'])
        print(df_coeff)
        return model

    # predict on test dataset
    def predict(self, model,  x_test, y_test):
        test_y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, test_y_pred)
        r2score = r2_score(y_test, test_y_pred)
        return mse, r2score

    # draw plots between independent and dependent attribute
    def drawplot(self, x_test, y_test):

        plt.subplot(331)
        plt.scatter(x_test[:, 0], y_test, s=10, color='r', marker=">")
        plt.xlabel('Cylinders', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(332)
        plt.scatter(x_test[:, 1], y_test, s=10, color='g', marker=(5, 0))
        plt.xlabel('Displacement', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())

        verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]])
        plt.subplot(333)
        plt.scatter(x_test[:, 2], y_test, s=10, color='b', marker=verts)
        plt.xlabel('Horsepower', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(334)
        plt.scatter(x_test[:, 3], y_test, s=10, color='y', marker=(5, 1))
        plt.xlabel('Weight', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(335)
        plt.scatter(x_test[:, 4], y_test, s=10, marker='+')
        plt.xlabel('Acceleration', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(336)
        plt.scatter(x_test[:, 5], y_test, s=10, color='r', marker=(5, 2))
        plt.xlabel('Model Year', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(337)
        plt.scatter(x_test[:, 6], y_test, s=10, color='b', marker=(5, 2))
        plt.xlabel('Origin', fontsize=10)
        plt.ylabel('MPG', fontsize=10)
        plt.grid(True)
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == "__main__":
    url = "https://s3.amazonaws.com/cs6375-datasets/datasett.csv"
    lr = LinearRegression(url)
    x_train, x_test, y_train, y_test = lr.preProcess()
    model = lr.train(x_train, y_train)
    mse_train, r2score_train = lr.predict(model, x_train, y_train)
    mse_test, r2score_test = lr.predict(model, x_test, y_test)
    print("Mean square error on train data:", mse_train)
    print("Mean square error on test data:", mse_test)
    print("Variance on test data:", r2score_test)
    lr.drawplot(x_test, y_test)

    #Residual plot
    plt.scatter(model.predict(x_train), model.predict(x_train) - y_train, c='b', s=20, alpha=0.5)
    plt.scatter(model.predict(x_test), model.predict(x_test) - y_test, c='r', s=20, alpha=0.5)
    plt.hlines(y=0, xmin=-2, xmax=2)
    plt.ylabel('Residuals', fontsize=10)
    plt.title("Residual plot using training data(blue) and test data(red)")
    plt.show()
