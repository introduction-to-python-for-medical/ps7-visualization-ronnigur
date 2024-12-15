import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd

# Fetch the diabetes dataset
data = fetch_openml(name='diabetes', version=1, as_frame=True)
print(data.DESCR)

# Load the data into a DataFrame
df = data.frame

# Display sample data and general statistics
print("Sample data:\n", df.sample(5))
print("\nSummary statistics:\n", df.describe())
print("\nData types:\n", df.dtypes)

# Select features for analysis
features = df.columns
selected_features = [features[0], features[2], features[4], features[6], features[7]]
print("Available features:", list(features))
print("Selected features:", selected_features)

# Create scatter plots for selected features
reference_feature = selected_features[1]
y = df[reference_feature]

fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))

for ax, f in zip(axs, selected_features):
    ax.scatter(df[f], y)
    ax.set_xlabel(f)
    ax.set_ylabel(reference_feature)

plt.tight_layout()
plt.show()

# Create histograms for selected features
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))

for ax, f in zip(axs, selected_features):
    ax.hist(df[f], bins=5, color='skyblue', edgecolor='black')
    ax.set_xlabel(f)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()

# Create a scatter plot for comparison between two features
reference_feature = selected_features[0]  # Reference feature
comparison_feature = selected_features[1]  # Feature to compare

plt.figure(figsize=(8, 6))
plt.scatter(df[reference_feature], df[comparison_feature], alpha=0.6, color='purple')
plt.xlabel(reference_feature)
plt.ylabel(comparison_feature)
plt.title(f"Scatter Plot: {reference_feature} vs {comparison_feature}")

# Save the scatter plot as an image file
plt.savefig('correlation_plot.png')
plt.show()


# Save analysis to a text file
with open("analysis.txt", "w") as file:
    file.write(analysis)

print("Analysis saved in 'analysis.txt'.")

