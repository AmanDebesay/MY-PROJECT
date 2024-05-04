import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import platform
from sklearn.linear_model import LogisticRegression
import tensorflow
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt

My_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
if not platform.system() == 'Windows':
    X_train, y_train = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')

else:
    X_train, y_train = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte')
if not platform.system() == 'Windows':
    X_test, y_test = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte',
        labels_path='t10k-labels.idx1-ubyte')

else:
    X_test, y_test = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte',
        labels_path='t10k-labels.idx1-ubyte')
# Display a sample data in index 1
test = X_test[1]
test = test.reshape([28, 28])
plt.gray()
plt.imshow(test)
plt.show()


# Train and evaluate a multiclass classifier using logistic regression algolithm

# Set regularization rate
reg = 0.1

# train a logistic regression model on the training set
multi_model = LogisticRegression(C=1 / reg, solver='lbfgs', multi_class='auto', max_iter=25).fit(X_train, y_train)
print(multi_model)

# predict
Image_predictions = multi_model.predict(X_test)
print('Predicted labels: ', Image_predictions[:15])
print('Actual labels   : ', y_test[:15])

# look at the classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, Image_predictions))

# You can get the overall metrics separately from the report using the scikit-learn metrics score classes, but with multiclass results you must
# specify which average metric you want to use for precision and recall
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Overall Accuracy:", accuracy_score(y_test, Image_predictions))
print("Overall Precision:", precision_score(y_test, Image_predictions, average='macro'))
print("Overall Recall:", recall_score(y_test, Image_predictions, average='macro'))

# look at the confusion matrix for our model:
from sklearn.metrics import confusion_matrix

# Print the confusion matrix
mcm = confusion_matrix(y_test, Image_predictions)
print(mcm)

print('\n>>> END <<<')

