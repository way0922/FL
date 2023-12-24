import pandas as pd
import numpy as np
import os

# Set the path for your input dataset and output location
input_path = 'D:/flower_test/data_10000/minmax-train - radom_onehot_label.csv'
output_folder = 'D:/flower_test/Quantity skew/beta=21'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
dataset = pd.read_csv(input_path)

# Define the number of clients and the beta parameter for quantity skew
num_clients = 5
beta = 21

# Ensure the dataset has the 'Label_' columns
label_columns = [col for col in dataset.columns if col.startswith('Label_')]

# Calculate the number of records each client should get
total_records = len(dataset)
records_per_client = np.random.dirichlet([beta] * num_clients) * total_records

# Round the records to integers
records_per_client = np.round(records_per_client).astype(int)

# Create a directory for each client
for i in range(num_clients):
    client_folder = os.path.join(output_folder, f'client_{i + 1}')
    os.makedirs(client_folder, exist_ok=True)

    # Sample the records for the current client
    sampled_records = dataset.sample(n=records_per_client[i], replace=False, random_state=i)

    # Save the sampled records to the client's folder
    output_path = os.path.join(client_folder, f'Quantity skew_data_client_{i + 1}.csv')
    sampled_records.to_csv(output_path, index=False)

print("Data distribution with quantity skew completed.")
