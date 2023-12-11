import pandas as pd
import numpy as np

# Load the dataset
file_location = 'D:/flower_test/data_10000/minmax-train - radom_onehot_label.csv'
df = pd.read_csv(file_location)

# Set the random seed for reproducibility
np.random.seed(42)

# Number of clients
num_clients = 5

# Set the value of K
K = 3  # Change this value according to your requirements

# Initialize the assigned_clients dictionary to store assigned clients for each label
assigned_clients = {f'Label_{i}': [] for i in range(1, 26)}

# Ensure each label is assigned to exactly K clients
for label in assigned_clients:
    # Randomly shuffle the list of clients and take the first K
    assigned_clients[label] = np.random.choice(range(1, num_clients + 1), size=K, replace=False)

# Save the assigned clients to a CSV file
assigned_clients_df = pd.DataFrame.from_dict(assigned_clients, orient='index').transpose()
# assigned_clients_df.to_csv('D:/flower_test/test/assigned_clients.csv', index=False)

# Save data for each assigned client
for client in range(1, num_clients + 1):
    client_df = pd.DataFrame()
    for label in assigned_clients:
        label_data = df[df[label] == 1]

        # Ensure label_data is not empty
        if not label_data.empty:
            num_records_for_client = len(label_data) // K
            remainder = len(label_data) % K

            # Find the indices of client occurrences in the assigned clients array
            indices = np.where(assigned_clients[label] == client)[0]

            if indices.size > 0:
                start_index = indices[0] * num_records_for_client + min(indices[0], remainder)
                end_index = start_index + num_records_for_client + int(indices[0] < remainder)

                client_df = pd.concat([client_df, label_data.iloc[start_index:end_index]])

    client_df.to_csv(f'D:/flower_test/Quantity_base label imbalance_data/client_{client}_data.csv', index=False)
