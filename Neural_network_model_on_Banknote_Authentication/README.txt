## Details of the Dataset

For the Neural Network problem given, I have used the [Banknote Authentication dataset](http://archive.ics.uci.edu/ml/datasets/banknote+authentication) from UCI machine learning repository. The Dataset was read from a [URL](http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt).
Data Attributes:-
1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous) 
3. curtosis of Wavelet Transformed image (continuous) 
4. entropy of image (continuous) 
5. class (integer) 
 

## PreProcessing

The dataset was split using the train_test_split imported from the sklearn.model_selection and test size was chosen to be 0.25.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=np.random)
```

## Compiling the Code
To compile the code just run
```python 
python ScikitLearnLab_part2.py
```