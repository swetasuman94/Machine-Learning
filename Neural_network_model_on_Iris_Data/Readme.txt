1. All the theory part is attached as a pdf file
2. The source code is available in a file name  NeuralNet.py
3. I've used Iris Dataset.
4. The dataset is loaded from the URL and it is preprocessed in which it is normalized and split into train and test data sets.
5. Make sure to have all the dependencies and libraries installed and simply run using command python NeuralNet.py. I've written and tested the code on mac.



Output for my dataset summarized in a tabular format for different combination of parameters



+------------------------+---------------------+-------------------+--------------------+
|                        | iterations - 5000   | iterations - 1000 | iterations - 500   |
+------------------------+---------------------+-------------------+--------------------+
|                        | test error          | test error        | test error         |
|                        | -------------       | -------------     | -------------      |
|                        |                     |                   |                    |
| learning rate - 0.0095 | Sigmoid - 0.0083    | Sigmoid - 3.109   | Sigmoid - 4.169    |
|                        |                     |                   |                    |
|                        | tanh - 0.0017       | tanh - 0.0147     | tanh - 0.0325      |
|                        |                     |                   |                    |
|                        | ReLu - 12.5         | ReLu - 12.5       | ReLu - 12.5        |
+------------------------+---------------------+-------------------+--------------------+
|                        | test error          | test error        | test error         |
|                        | -------------       | -------------     | -------------      |
|                        |                     |                   |                    |
| learning rate - 0.002  | Sigmoid - 2.7161    | Sigmoid - 4.249   | Sigmoid - 4.266    |
|                        |                     |                   |                    |
|                        | tanh - 0.0173       | tanh - 0.097      | tanh - 0.1285      |
|                        |                     |                   |                    |
|                        | ReLu - 12.5         | ReLu - 12.5       | ReLu - 12.5        |
+------------------------+---------------------+-------------------+--------------------+
|                        | test error          | test error        | test error         |
|                        | -------------       | -------------     | -------------      |
|                        |                     |                   |                    |
| learning rate - 0.09   | Sigmoid - 0.000531  | Sigmoid - 0.0036  | Sigmoid - 0.00989  |
|                        |                     |                   |                    |
|                        | tanh - 6.4999       | tanh - 6.4999     | tanh - 6.4999      |
|                        |                     |                   |                    |
|                        | ReLu - 12.5         | ReLu - 12.5       | ReLu - 12.5        |
+------------------------+---------------------+-------------------+--------------------+



-->From above table of Results I can say that:

1.while keeping a constant learning rate if you increase the number of max iterations then test error decreases for sigmoid, tanh and doesn't change for ReLu.
2.while keeping the number of max iterations constant if you increase the learning rate then test error is varying without consistency 
3.The least test error is obtained for sigmoid function with parameters as iterations-5000 and learning_rate-0.09.

--May be the reason for sigmoid giving least test error is, that the output from sigmoid is unnormalized probability, which simplifies the classification task.