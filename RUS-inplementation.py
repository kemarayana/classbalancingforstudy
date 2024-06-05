import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'final_combined_data_with_readmission.csv'
data = pd.read_csv(file_path)

# Display the original distribution of the 'readmission' column
print("Original distribution:")
print(data['readmission'].value_counts())

# Plot the original distribution
plt.figure(figsize=(10, 5))
data['readmission'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Original Class Distribution')
plt.xlabel('Readmission')
plt.ylabel('Count')
plt.show()

# Separate the features and the target
X = data.drop(columns=['readmission'])
y = data['readmission']

# Define the undersampling strategy
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Apply the undersampling strategy
X_resampled, y_resampled = rus.fit_resample(X, y)

# Combine the resampled features and target into a single DataFrame
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

# Display the new distribution of the 'readmission' column
print("Resampled distribution:")
print(resampled_data['readmission'].value_counts())

# Plot the resampled distribution
plt.figure(figsize=(10, 5))
resampled_data['readmission'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Resampled Class Distribution')
plt.xlabel('Readmission')
plt.ylabel('Count')
plt.show()

# Save the resampled data to a new CSV file (optional)
resampled_data.to_csv('resampled_data-v.csv', index=False)

# Display the first few rows of the resampled data
print(resampled_data.head())
