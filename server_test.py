'''
引入套件
'''
from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import tensorflow as tf # 建立Global model並取得初始參數
from tensorflow.keras import Input, Model, layers, models # 建立CNN架構
import numpy as np # 資料前處理
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import recall_score
import csv,os
'''
Step 1. Build Global Model (建立全域模型)
'''
# Hyperparameter超參數


dftest = pd.read_csv("D:/flower_test/data_10000/minmax-test - radom(修改ebay).csv")


x_columns = dftest.columns.drop(dftest.filter(like='Label_').columns)
x_test = dftest[x_columns].values.astype('float32')
y_test = dftest.filter(like='Label_').values.astype('float32')





input = 1275
# Build Model
def MLP_Model(input):
  
  It = tf.random_normal_initializer()
  model = tf.keras.Sequential()
  model.add(Dense(256, input_dim=input, activation='relu',kernel_initializer=It))
  model.add(Dense(256, activation='relu',kernel_initializer=It))
  model.add(Dropout(0.2))
  #model.add(Dense(256, activation='relu'))
  #model.add(Dropout(0.2))
  model.add(Dense(25, activation='softmax',kernel_initializer=It))

  return model
 

'''
Step 2. Start server and run the strategy (套用所設定的策略，啟動Server)
'''

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = MLP_Model(input)
    #model.summary()
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    csv_filename = 'D:/flower_test/server_accuracy/evaluation_results.csv'
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5, # 每一輪參與Training的Client比例
        fraction_eval=0.5, # 每一輪參與Evaluating的Client比例
        min_fit_clients=2, # 每一輪參與Training的最少Client連線數量 (與比例衝突時,以此為準)
        min_eval_clients=2, # 每一輪參與Evaluating的最少Client連線數量 (與比例衝突時,以此為準)
        min_available_clients=2, # 啟動聯合學習之前，Client連線的最小數量
        on_fit_config_fn=fit_config, # 設定 Client-side Training Hyperparameter  
        on_evaluate_config_fn=evaluate_config, # 設定 Client-side Evaluating Hyperparameter
        eval_fn=get_eval_fn(model, x_test, y_test,csv_filename), # 設定 Server-side Evaluating Hyperparameter (用Global Dataset進行評估)
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()), # Global Model 初始參數設定
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(server_address="0.0.0.0:8080", config={"num_rounds": 5}, strategy=strategy) #windows

'''
[Model Hyperparameter](Client-side, train strategy)
* 設定Client Training 的 Hyperparameter: 包含batch_size、epochs、learning-rate...皆可設定。
* 甚至可以設定不同 FL round 給予 client 不同的 Hyperparameter
'''
def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 128, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 128,
        "local_epochs": 1 if rnd < 2 else 2, # Client 進行 local model Training時，前兩輪的epoch設為1，之後epoch設為2
    }
    return config

'''
[Model Hyperparameter](Client-side, evaluate strategy)
* 設定Client Testing 的 Hyperparameter: 包含epochs、steps(Total number of steps, 也就是 batche個數 (batches of samples))。
* 可以設定不同 FL round 給予 client 不同的 Hyperparameter
'''
def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10 # Client 進行 local model evaluate時，前4輪 step 為 5，之後 step 為 10
    return {"val_steps": val_steps}

'''
[Model Hyperparameter](Server-side, evaluate strategy) 
用 Global Dataset 評估 Global model (不含訓練)
'''


def get_eval_fn(model, x_test, y_test, csv_filename):
    # Initialize a counter for the round number
    round_counter = 0

    # Open the CSV file in append mode and create a CSV writer
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Round', 'Loss', 'Accuracy', 'Recall_weighted', 'Recall_macro']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header if the file is empty
        if csvfile.tell() == 0:
            csv_writer.writeheader()

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        nonlocal round_counter
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test)

        # Predict on the test set
        y_pred = model.predict(x_test)

        # Convert one-hot encoded labels to integers
        y_true_int = np.argmax(y_test, axis=1)
        y_pred_int = np.argmax(y_pred, axis=1)

        # Calculate recall
        recall = recall_score(y_true_int, y_pred_int, average='weighted')
        recall1 = recall_score(y_true_int, y_pred_int, average='macro')

        # Increment the round counter
        round_counter += 1

        # Append the results to the CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writerow({
                'Round': round_counter,
                'Loss': loss,
                'Accuracy': accuracy,
                'Recall_weighted': recall,
                'Recall_macro': recall1
            })

        print(f'Recall: {recall}')
        return loss, {"accuracy": accuracy, "recall_weighted": recall, "recall_macro": recall1}

    return evaluate


'''
main
'''
if __name__ == "__main__":
    main()