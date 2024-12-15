import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Load the dataset
data = fetch_openml(name='diabetes', version=1, as_frame=True)
df = data.frame

# Print dataset description
print(data.DESCR)

# Features selection
features = list(df.columns)
selected_features = features[:9]  # Select the first 9 features
print("Available features:", features)
print("Selected features: ", selected_features)

# Distribution histograms
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))
for ax, f in zip(axs, selected_features):
    ax.hist(df[f], bins=5, color='skyblue', edgecolor='black')
    ax.set_xlabel(f)

# Scatter plots against reference feature
reference_feature = selected_features[3]  # Reference feature
y = df[reference_feature]

fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))
for ax, f in zip(axs, selected_features):
    ax.scatter(df[f], y, alpha=0.6)
    ax.set_xlabel(f)

plt.show()

# Correlation plot between two selected features
comparison_feature = selected_features[5]
plt.figure(figsize=(8, 6))
plt.scatter(df[reference_feature], df[comparison_feature], alpha=0.6)
plt.xlabel(reference_feature)
plt.ylabel(comparison_feature)

# Compute Pearson correlation coefficient
correlation = df[reference_feature].corr(df[comparison_feature])
print(f"Correlation between {reference_feature} and {comparison_feature}: {correlation:.4f}")

# Save the plot as an image
plt.savefig('correlation_plot.png')
plt.show()
