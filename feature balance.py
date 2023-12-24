import pandas as pd

# Load the datasets
train_data = pd.read_csv('D:/flower_test/5G dataset onehot/5Gtrain_dataset_onehot.csv')
test_data = pd.read_csv('D:/flower_test/5G dataset onehot/5Gtest_dataset_onehot.csv')

# Find the set of unique features in each dataset
train_features = set(train_data.columns)
test_features = set(test_data.columns)

# Identify features present in the training dataset but not in the test dataset
additional_features_train = train_features - test_features

# Identify features present in the test dataset but not in the training dataset
additional_features_test = test_features - train_features

# Add the missing features to the test dataset with default values (you may need to choose appropriate default values)
for feature in additional_features_train:
    test_data[feature] = 0  # You can use any default value

# Add the missing features to the training dataset with default values
for feature in additional_features_test:
    train_data[feature] = 0  # You can use any default value

# Now both datasets have the same set of features
# Save the updated datasets if needed
train_data.to_csv('D:/flower_test/5G dataset onehot/updated_5Gtrain_dataset_onehot.csv', index=False)
test_data.to_csv('D:/flower_test/5G dataset onehot/updated_5Gtest_dataset_onehot.csv', index=False)
