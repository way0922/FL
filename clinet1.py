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

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

gobal_loss = []
gobal_val_loss = []
gobal_accuracy = []
gobal_val_accuracy = []



dftrain = pd.read_csv("D:/flower/50,30,20/train50%.csv")
dftest = pd.read_csv("./data_10000/minmax-test - radom(修改ebay).csv")


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

# Define your neural network model
model = tf.keras.Sequential()
model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='softmax'))

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_test, y_test):
        super().__init__()
        self.client_id = client_id
        self.client_loss = []
        self.client_accuracy = []
        self.client_metrics = {"precision": [], "recall": [], "f1_score": []}
        self.x_test = x_test
        self.y_test = y_test
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train,validation_data=(x_val, y_val),epochs=50, batch_size=4096)
        gobal_loss.extend(history.history['loss'])
        gobal_val_loss.extend(history.history['val_loss'])
        gobal_accuracy.extend(history.history['accuracy'])
        gobal_val_accuracy.extend(history.history['val_accuracy'])

        self.client_loss.extend(history.history['loss'])
        self.client_accuracy.extend(history.history['accuracy'])

        return model.get_weights(), len(x_train), {}

def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(self.x_test, self.y_test)
    num_examples_test = len(self.x_test)

    y_test_pred = model.predict(self.x_test)
    y_test_pred_class = np.argmax(y_test_pred, axis=1)
    recall_values = []

    for class_idx in range(25):
        true_labels = (self.y_test == class_idx)
        predicted_labels = (y_test_pred_class == class_idx)
        precision = precision_score(true_labels, predicted_labels, average='None')
        recall = recall_score(true_labels, predicted_labels, average='None')
        f1 = f1_score(true_labels, predicted_labels, average='None')

        recall_values.append(recall)
        self.client_metrics["precision"].append(precision)
        self.client_metrics["recall"].append(recall)
        self.client_metrics["f1_score"].append(f1)

        print(f"Client {self.client_id}, Label {class_idx} - Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

    self.plot_recall_curve(recall_values)    
    return loss, num_examples_test, {"accuracy": accuracy, "client_metrics": self.client_metrics}
def plot_recall_curve(self, recall_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(25), recall_values, marker='o')
    plt.title('Recall Curve for Each Label')
    plt.xlabel('Label Index')
    plt.ylabel('Recall')
    plt.savefig(f'recall_curve_client_{self.client_id}.png')

# Start Flower client
client = CifarClient(client_id=0, x_test=x_test, y_test=y_test)
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# Access the loss and accuracy history


# Plot the loss

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(gobal_loss, label='Training Loss')
plt.plot(gobal_val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the accuracy
plt.subplot(1, 2, 2)
plt.plot(gobal_accuracy, label='Training Accuracy')
plt.plot(gobal_val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.gcf().canvas.set_window_title('Clinet1')
plt.tight_layout()
plt.show()
# Make predictions on the test data
y_test_pred = model.predict(x_test)

# Convert one-hot encoded labels back to class labels for y_test
y_test_class = np.argmax(y_test, axis=1)

# Convert predicted probabilities to class labels
y_test_pred_class = np.argmax(y_test_pred, axis=1)

# Define true and predicted labels
true_labels = y_test_class
predicted_labels = y_test_pred_class

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)


# Define a dictionary mapping class indices to class names
class_names = {
    0:'AppleiCloud',
    1: 'AppleiTunes',
    2:'Dropbox',
    3:'FTP_DATA',
    4:'Facebook',
    5:'GMail',
    6:'Github',
    7:'GoogleDrive',
    8:'GoogleHangoutDuo',
    9:'GoogleServices',
    10:'Instagram',
    11:'MS_OneDrive',
    12:'NetFlix',
    13:'Skype',
    14:'Snapchat',
    15:'SoundCloud',
    16:'Spotify',
    17:'Steam',
    18:'TeamViewer',
    19:'Telegram',
    20:'Twitter',
    21:'WhatsApp',
    22:'Wikipedia',
    23:'YouTube',
    24:'eBay'
    # Add mappings for all 25 classes here
}



f1_scores = []
precision_scores = []
recall_scores = []
label_counts = {label: (y_test_class == label).sum() for label in range(25)}

for class_idx in range(25):  # Assuming 25 classes
    true_labels = (y_test_class == class_idx)
    predicted_labels = (y_test_pred_class == class_idx)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)


results_table = pd.DataFrame({
    'Class': [class_names[i] for i in range(25)],
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores,
    'Label Count': [label_counts[i] for i in range(25)]
    
})

# Print the results table
print("Results Table:")
print(results_table)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [class_names[i] for i in classes], rotation=45)
    plt.yticks(tick_marks, [class_names[i] for i in classes])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()