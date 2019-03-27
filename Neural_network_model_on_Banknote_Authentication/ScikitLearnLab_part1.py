
from sklearn.neural_network import MLPClassifier

class NeuralNet:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # Below is the training function
    def train(self):
        model = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(2), random_state=1, max_iter=1000)
        model.fit(self.X, self.Y)
        return model

    # predict on test dataset
    def predict(self, model):
        Y_predict = model.predict(self.X)
        print("Y predicted by model :", Y_predict)
        print("Intercepts for neurons :", model.intercepts_)
        print("Weights for neurons :", model.coefs_)

if __name__ == "__main__":
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y = [0, 1, 1, 0]
    neural_network = NeuralNet(X, Y)
    model = neural_network.train()
    neural_network.predict(model)