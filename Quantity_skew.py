import pandas as pd
import numpy as np
import os

# Set the data set location
data_path = 'D:/flower_test/data_10000/minmax-train - radom_onehot_label.csv'

# Set the output location
output_path = 'D:/flower_test/Quantity skew/beta=21'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load the dataset
dataset = pd.read_csv(data_path)

# Define the number of clients
num_clients = 5

# Define the skewness parameter (beta)
beta = 21

# Function to distribute data with quantity skew
def distribute_quantity_skew(data, num_clients, beta):
    # Calculate the total number of records
    total_records = len(data)

    # Initialize empty lists to store data for each client
    clients_data = [[] for _ in range(num_clients)]

    # Define weights for each client
    client_weights = np.random.dirichlet(np.ones(num_clients) * beta)

    # Iterate over labels starting with "Label_"
    label_columns = [col for col in data.columns if col.startswith("Label_")]
    for label_column in label_columns:
        # Calculate the number of records for each client based on the label count
        label_data = data[data[label_column] == 1]
        label_count = len(label_data)

        # Calculate the target number of records for each client for this label
        target_counts = np.floor(client_weights * label_count).astype(int)

        # Adjust one client's count to match the total number of records for this label
        target_counts[-1] += label_count - sum(target_counts)

        # Distribute the records directly among the clients with skew for this label
        for i in range(num_clients):
            client_records = target_counts[i]

            # Add records to the corresponding client's data
            clients_data[i].extend(label_data.sample(client_records, replace=True).index)

    # Concatenate data for each client
    for i in range(num_clients):
        clients_data[i] = data.loc[clients_data[i]]

    # Ensure the total number of records is the same before and after distribution
    total_records_after_distribution = sum(len(client) for client in clients_data)
    print(f"Total Records Before Distribution: {total_records}")
    print(f"Total Records After Distribution: {total_records_after_distribution}")
    assert total_records == total_records_after_distribution, "Total records mismatch after distribution"

    return clients_data

# Distribute data with quantity skew
clients_data = distribute_quantity_skew(dataset, num_clients, beta)

# Save data for each client
for i in range(num_clients):
    output_file = os.path.join(output_path, f'client_{i + 1}_data.csv')
    clients_data[i].to_csv(output_file, index=False)
