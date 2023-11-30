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

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

gobal_loss = []
gobal_val_loss = []
gobal_accuracy = []
gobal_val_accuracy = []



dftrain = pd.read_csv("D:/flower_test/client_data/beta=0.1/client_3.csv")
dftest = pd.read_csv("D:/flower_test/data_10000/minmax-test - radom(修改ebay).csv")


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
model.add(Dense(25, activation='softmax',kernel_initializer=It))

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define Flower client
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
        history = model.fit(x_train, y_train,validation_data=(x_val, y_val),epochs=50, batch_size=512)
        gobal_loss.extend(history.history['loss'])
        gobal_val_loss.extend(history.history['val_loss'])
        gobal_accuracy.extend(history.history['accuracy'])
        gobal_val_accuracy.extend(history.history['val_accuracy'])

        self.client_loss.extend(history.history['loss'])
        self.client_accuracy.extend(history.history['accuracy'])

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        

        y_pred = model.predict(self.x_test)
        y_test_class = np.argmax(y_test, axis=1)
        y_test_pred_class = np.argmax(y_pred, axis=1)
        true_labels = y_test_class
        predicted_labels = y_test_pred_class
        class_names = {
            0:'AppleiCloud',1: 'AppleiTunes',2:'Dropbox',3:'FTP_DATA',4:'Facebook',5:'GMail',6:'Github',7:'GoogleDrive',8:'GoogleHangoutDuo',9:'GoogleServices',10:'Instagram',11:'MS_OneDrive',12:'NetFlix',13:'Skype',14:'Snapchat',15:'SoundCloud',16:'Spotify',17:'Steam',18:'TeamViewer',19:'Telegram',20:'Twitter',21:'WhatsApp',22:'Wikipedia',23:'YouTube',24:'eBay'
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
        recall = recall_score(np.argmax(y_test, axis=1),y_test_pred_class, average='macro')
        loss,accuracy = model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)
        model.set_weights(parameters)
        return loss,num_examples_test,{"recall": recall,"accuracy":accuracy}

# Start Flower client
client = CifarClient(client_id=3, x_test=x_test, y_test=y_test)
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









