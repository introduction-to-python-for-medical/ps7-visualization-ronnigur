# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Fetch the dataset
data = fetch_openml(name='diabetes', version=1, as_frame=True)
print(data.DESCR)

# Load the dataset into a DataFrame
df = data.frame

# Display a random sample of rows and data types
print(df.sample(5))
print(df.dtypes)

# Select all features from the dataset
features = list(df.columns)
selected_features = [features[0], features[1], features[2], features[3],features[4], features[5], features[6], features[7], features[8]]

print("Available features:", features)
print("Selected features: ", selected_features)

# Plot histograms for the selected features
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))

for ax, f in zip(axs, selected_features):
    ax.hist(df[f], bins=5, color='skyblue', edgecolor='black')
    ax.set_xlabel(f)

plt.tight_layout()
plt.show()

# Select a reference feature for scatter plots
reference_feature = selected_features[3]  # Choosing the 4th feature (index 3)
y = df[reference_feature]

# Create scatter plots of selected features against the reference feature
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))

for ax, f in zip(axs, selected_features):
    ax.scatter(df[f], y, alpha=0.6, color='blue')
    ax.set_xlabel(f)
    ax.set_ylabel(reference_feature)

plt.tight_layout()
plt.show()

# Create a scatter plot for a specific pair of features
reference_feature = selected_features[3]  # The reference feature
comparison_feature = selected_features[5]  # A feature to compare to

plt.figure(figsize=(8, 6))
plt.scatter(df[reference_feature], df[comparison_feature], alpha=0.6, color='green')
plt.xlabel(reference_feature)
plt.ylabel(comparison_feature)
plt.title(f'Scatter plot of {reference_feature} vs {comparison_feature}')

# Save the plot as an image file
plt.savefig('correlation_plot.png')
plt.show()
