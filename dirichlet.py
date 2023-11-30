import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import os 

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = np.max(train_labels) + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

    # Set your desired values
n_clients = 5
dirichlet_alpha = 0
seed = 42
csv_file_path = 'D:/flower/data_10000/minmax-train - radom_onehot_label.csv'  # Path to your CSV file
output_dir = 'D:/flower/client_data/beta=0'  # Output directory to save client data
if __name__ == "__main__":
    np.random.seed(seed)

    # Load your CSV file using pandas
    data = pd.read_csv(csv_file_path)

    # Assuming your CSV file has features and labels
    features = data.drop(data.columns[data.columns.str.startswith('Label_')], axis=1)
    labels = data[data.columns[data.columns.str.startswith('Label_')]]

    # Convert features and labels to numpy arrays
    train_images = np.array(features)
    train_labels = np.array(labels)

    # Continue with the rest of the code as before
    classes = np.arange(train_labels.shape[1])  # Assuming one-hot encoded labels
    n_classes = len(classes)

    # Split the dataset into non-IID distribution for clients
    client_idcs = dirichlet_split_noniid(np.argmax(train_labels, axis=1), alpha=dirichlet_alpha, n_clients=n_clients)

    for i, client_indices in enumerate(client_idcs):
        client_data = data.iloc[client_indices]
        client_filename = os.path.join(output_dir, f'client_{i + 1}.csv')
        client_data.to_csv(client_filename, index=False)
        print(f"Saved client {i + 1} data to {client_filename}")

    # Display label distribution on different clients
    plt.figure(figsize=(12, 8))
    plt.hist([np.argmax(train_labels[idc], axis=1) for idc in client_idcs], stacked=True,
             bins=np.arange(np.min(classes) - 0.5, np.max(classes) + 1.5, 1),
             label=["Client {}".format(i) for i in range(n_clients)],
             rwidth=0.5)
    plt.xticks(np.arange(n_classes), classes)
    plt.xlabel("Label type")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    #plt.title("Display Label Distribution on Different Clients")
    plt.title("Distribution-based label imbalance(beta=5)")
    
    plt.show()
