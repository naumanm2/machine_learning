import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data = pd.read_csv("datasets/train.csv").to_numpy()
test_data = pd.read_csv("datasets/test.csv").to_numpy()

data_x = data[0:40000, 1:]
data_y = data[0:40000, 0]

x_test = test_data[600:1000]

x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2)

n_of_neighbors = range(1, 9)

val_scores = []
train_scores = []


for n in n_of_neighbors:
    print("k = " + str(n) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train, y_train)
    train_pred = knn.predict(x_train)
    y_pred = knn.predict(x_val)
    train_accuracy = knn.score(x_train, y_train)
    accuracy = accuracy_score(y_val,y_pred)
    val_scores.append(accuracy)
    train_scores.append(train_accuracy)
    end = time.time()
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    
    print("Complete time: " + str(end-start) + " secs.")
    
print("KNN training scores: ", train_scores)
print("KNN validation scores: ", val_scores)


test_pred = knn.predict(x_test)

nrows=4
ncols=4



_, ax = plt.subplots(nrows, ncols)
for row in range(nrows):
    for col in range(ncols):
        ax[row, col].set_axis_off()
        image = x_test[(row+1)*(col+1)].reshape(28, 28)
        ax[row, col].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax[row, col].set_title(f'Prediction: {test_pred[(row+1)*(col+1)]}')
    


#plt.plot(n_of_neighbors, scores, color='red')
#plt.plot(n_of_neighbors, train_scores, color='green')
#plt.xlabel('Value of K')
#plt.ylabel('Testing accuracy')
plt.show()
    

