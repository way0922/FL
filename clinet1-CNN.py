'''
引入套件
'''
from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import tensorflow as tf # 建立Global model並取得初始參數
from tensorflow.keras import Input, Model, layers, models # 建立CNN架構
import numpy as np # 資料前處理

'''
Step 1. Build Global Model (建立全域模型)
'''
# Hyperparameter超參數
num_classes = 10
input_shape = (28, 28, 1)

# Build Model
def CNN_Model(input_shape, number_classes):
    # define Input layer
    input_tensor = Input(shape=input_shape) # Input: convert normal numpy to Tensor (float32)

    # define layer connection
    x = layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu")(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(number_classes, activation="softmax")(x)

    # define model
    model = Model(inputs=input_tensor, outputs=outputs, name="mnist_model")
    return model

'''
Step 2. Override fl.server.strategy.FedAvg 覆寫FedAvg class
'''
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:

        # [!!!] 此處再次使用原本 fl.server.strategy.FedAvg 的 aggregate_fit，
        # aggregate_fit() 會回傳聚合後的模型權重參數
        # 相關官方文件: #https://flower.dev/docs/_modules/flwr/server/strategy/fedavg.html#FedAvg.aggregate_fit
        # [!!!] 若要改寫聚合演算法，此處就不能寫 aggregated_weights = super().aggregate_fit(rnd, results, failures)
        # 而是要參考文件中 aggregate_fit() 的程式碼，去改寫聚合演算法
        aggregated_weights = super().aggregate_fit(rnd, results, failures) 

        # 取出 Ckient 回傳的結果： https://flower.dev/docs/_modules/flwr/common/typing.html#FitRes
        # accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        # type(r.metrics) = dict
        for _, r in results: # 取得 Client 回傳的訓練結果
            print(f"\n[!!!!!!!!] Client loss = { r.metrics['loss'] }")
            print(f"[!!!!!!!!] Client accuracy = { r.metrics['accuracy'] }")
            print(f"[!!!!!!!!] Client val_loss = { r.metrics['val_loss'] }")
            print(f"[!!!!!!!!] Client val_accuracy = { r.metrics['val_accuracy'] }\n")

        # 將Model權重參數存起來
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

'''
Step 3. Start server and run the strategy (套用所設定的策略，啟動Server)
'''

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = CNN_Model(input_shape=input_shape, number_classes=num_classes)
    #model.summary()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.5, # 每一輪參與Training的Client比例
        fraction_eval=0.5, # 每一輪參與Evaluating的Client比例
        min_fit_clients=2, # 每一輪參與Training的最少Client連線數量 (與比例衝突時,以此為準)
        min_eval_clients=1, # 每一輪參與Evaluating的最少Client連線數量 (與比例衝突時,以此為準)
        min_available_clients=2, # 啟動聯合學習之前，Client連線的最小數量

        on_fit_config_fn=fit_config, # 設定 Client-side Training Hyperparameter  
        on_evaluate_config_fn=evaluate_config, # 設定 Client-side Evaluating Hyperparameter
        eval_fn=get_eval_fn(model), # 設定 Server-side Evaluating Hyperparameter (用Global Dataset進行評估)
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()), # Global Model 初始參數設定
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080", config={"num_rounds": 2}, strategy=strategy) #windows

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
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data() # Train sample size: 60000

    # Data preprocessing
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[60000-5000:], y_train[60000-5000:]

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate

'''
main
'''
if __name__ == "__main__":
    main()