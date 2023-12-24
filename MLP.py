import os
import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import itertools
import seaborn as sns
import json


dftrain = pd.read_csv("D:/flower_test/5G train_test/5Gtrain_dataset_result.csv")
dftest = pd.read_csv("D:/flower_test/5G train_test/5Gtest_dataset_result.csv")


x_columns = dftest.columns.drop(dftest.filter(like='Label_').columns)
x_test = dftest[x_columns].values.astype('float32')
y_test = dftest.filter(like='Label_').values.astype('float32')




x_columns = dftrain.columns.drop(dftrain.filter(like='Label_').columns)
x_train = dftrain[x_columns].values.astype('float32')  
y_train = dftrain.filter(like='Label_').values.astype('float32')  

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=7, shuffle=True)


# Set the random seed for TensorFlow
tf.random.set_seed(7)

# Set the random seed for NumPy
np.random.seed(7)
It = tf.random_normal_initializer()
# Define your neural network model
model = tf.keras.Sequential()
model.add(Dense(256, input_dim=x_train.shape[1], activation='relu',kernel_initializer=It))
model.add(Dense(256, activation='relu',kernel_initializer=It))
model.add(Dropout(0.2))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(14, activation='softmax',kernel_initializer=It))

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, epochs=100, batch_size=512)
y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test, axis=1)
y_test_pred_class = np.argmax(y_pred, axis=1)
true_labels = y_test_class
class_names = {
            0:'AfreecaTV',1: 'Battleground',2:'GeForce_Now',3:'Google_Meet',4:'KT_GameBox',5:'MS_Teams',6:'Naver_NOW',7:'Netflix',8:'Roblox',9:'Teamfight_Tactics',10:'YouTube',11:'YouTube_Live',12:'Zepeto',13:'Zoom'
    # Add mappings for all 25 classes here
}
       
f1_scores = []
precision_scores = []
recall_scores = []
label_counts = {label: (y_test_class == label).sum() for label in range(14)}

for class_idx in range(14):
    class_true_labels = (y_test_class == class_idx)
    class_predicted_labels = (y_test_pred_class == class_idx)
    
    f1 = f1_score(class_true_labels, class_predicted_labels, zero_division=1)
    precision = precision_score(class_true_labels, class_predicted_labels, zero_division=1)
    recall = recall_score(class_true_labels, class_predicted_labels, zero_division=1)
    
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

# Rest of the code remains unchanged
results_table = pd.DataFrame({
    'Class': [class_names[i] for i in range(14)],
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores,
    'Label Count': [label_counts[i] for i in range(14)]
})

# Print the results table
print("Results Table:")
print(results_table)

  

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot the loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# ... (Previous code)

# Calculate the confusion matrix
conf_mat = confusion_matrix(true_labels, y_test_pred_class)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
accuracy = accuracy_score(y_test_class, y_test_pred_class)

print(f"Accuracy on the testing set: {accuracy:.4f}")
# Plot non-normalized confusion matrix
plt.figure(figsize=(8, 8))
plot_confusion_matrix(conf_mat, classes=class_names.values(), title='Confusion Matrix')

plt.show()







