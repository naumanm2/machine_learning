import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data = pd.read_csv("datasets/train.csv").to_numpy()
test_data = pd.read_csv("datasets/test.csv").to_numpy()

data_x = data[:30000, 1:]
data_y = data[:30000, 0]

test_data_x = data[30000:32000, 1:]
test_data_y = data[30000:32000, 0]

x_test = test_data[600:1000]

x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2)

scores = []
train_scores = []

clf = RandomForestClassifier(n_estimators=25)

#clf.fit(data_x, data_y)
#train_scores = cross_val_score(clf, x_train, y_train, cv=5)
#val_scores = cross_val_score(clf, x_val, y_val, cv=5)

scores = cross_validate(clf, data_x, data_y, cv=5, return_train_score=True)

train_scores = scores['train_score']
val_scores = scores['test_score']

clf.fit(data_x, data_y)

y_pred = clf.predict(test_data_x)
test_pred = clf.predict(x_test)

nrows=4
ncols=4

_, ax = plt.subplots(nrows, ncols)
for row in range(nrows):
    for col in range(ncols):
        ax[row, col].set_axis_off()
        image = x_test[(row+1)*(col+1)].reshape(28, 28)
        ax[row, col].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax[row, col].set_title(f'Prediction: {test_pred[(row+1)*(col+1)]}')


print("Random forest training scores: ", train_scores, "mean: ", train_scores.mean())
print("Random forest validation scores: ", val_scores, "mean: ", val_scores.mean())
print("fit time: ", scores['fit_time'], "\nscore time: ", scores['score_time'])
print(classification_report(test_data_y, y_pred))
print(confusion_matrix(test_data_y, y_pred))

plt.show()



