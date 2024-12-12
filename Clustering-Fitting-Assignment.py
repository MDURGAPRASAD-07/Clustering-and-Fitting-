#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[46]:


data = pd.read_csv(r"C:/Users/91910/OneDrive/Desktop/assignment_data/ICRISAT.csv")
data.head(10)


# In[5]:


# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Skewness and kurtosis for key columns
from scipy.stats import skew, kurtosis

print("\nSkewness and Kurtosis:")
numeric_columns = data.select_dtypes(include=['number']).columns  # Select only numeric columns
for col in numeric_columns:
    print(f"{col}: Skewness={skew(data[col], nan_policy='omit')}, Kurtosis={kurtosis(data[col], nan_policy='omit')}")

    
    


# # HISTOGRAM

# In[45]:


# Select numerical columns for clustering
numerical_columns = [
    "RICE AREA (1000 ha)", "RICE PRODUCTION (1000 tons)", "WHEAT AREA (1000 ha)",
    "WHEAT PRODUCTION (1000 tons)", "COTTON AREA (1000 ha)", "COTTON PRODUCTION (1000 tons)"
]

# Replace missing values in numerical columns with the column mean
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Standardize the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_columns])

# Perform K-Means clustering
num_clusters = 3  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Define a function to plot histogram with clusters
def plot_histogram_with_clusters(data, column_name):
    """
    Plots a histogram of a specified column, grouped by clusters.
    Ensures that the legend is ordered numerically by cluster.
    """
    if column_name not in data.columns or 'Cluster' not in data.columns:
        raise ValueError(f"Column '{column_name}' or 'Cluster' not found in dataset.")

    plt.figure(figsize=(10, 6))
    
    # Sort clusters before plotting
    sorted_clusters = sorted(data['Cluster'].unique())
    
    for cluster in sorted_clusters:
        cluster_data = data[data['Cluster'] == cluster][column_name]
        plt.hist(
            cluster_data,
            bins=10,
            alpha=0.6,
            label=f"Cluster {cluster}",
            edgecolor='black'
        )

    plt.title(f"Histogram of {column_name} by Clusters", fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(title='Cluster', loc='upper right')
    plt.show()

# Plot histogram for "RICE AREA (1000 ha)"
plot_histogram_with_clusters(data, 'RICE AREA (1000 ha)')


# # SCATTER PLOT

# In[40]:


# Define the regression and plotting function
def plot_regression(data, x_col, y_col):
    
    # Ensure required columns exist
    required_columns = [x_col, y_col]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    # Extracting data for regression
    X = data[[x_col]]
    y = data[y_col]

    # Ensuring there are no NaN values
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Input data contains NaN values. Please clean the data before regression.")

    # Creating and fitting the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting values for the regression line
    x_vals = np.linspace(X.min().values[0], X.max().values[0], 100)
    y_vals = model.predict(x_vals.reshape(-1, 1))

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, color='blue', alpha=0.7, s=100, edgecolor='k')

    # Adding regression line
    plt.plot(x_vals, y_vals, color='red', label='Regression Line')
    
    # Styling plots
    plt.title(f"Scatter Plot of {y_col} vs {x_col} with Regression Line", fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend(loc='best')
    plt.show()

    # Printing model details
    print(f"Regression Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")
    return model

# Apply the function to your dataset
model = plot_regression(data, 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)')


# # BOX PLOT

# In[37]:


# Select numerical columns for fitting
numerical_columns = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)']
selected_data = data[numerical_columns].dropna()

# Perform linear regression and create a boxplot
def create_boxplot_with_fitting(data, feature_x, feature_y):
    
    # Prepare data
    X = data[feature_x].values.reshape(-1, 1)
    y = data[feature_y]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict using the regression model
    predictions = model.predict(X)

    # Calculate residuals
    residuals = y - predictions

    # Add residuals to the data
    data['Residuals'] = residuals

    # Create a boxplot of residuals
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Residuals'], color='skyblue')
    plt.title(f'Boxplot of Residuals for {feature_y} vs {feature_x}', fontsize=16)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid()
    plt.show()

    # Print model performance metrics
    r2 = r2_score(y, predictions)
    print(f"R² Score: {r2:.4f}")
    print(f"Coefficients: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")

# Apply the function to create a boxplot
create_boxplot_with_fitting(selected_data, 'RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)')


# # ELBOW PLOT

# In[42]:


# Select numerical columns for clustering
numerical_columns = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)']
selected_data = data[numerical_columns].dropna()

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_data)

# Function to create the elbow plot
def plot_elbow(data):
  
    inertias = []
    for k in range(1, 11):  # Checking clusters from 1 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Plotting the elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertias, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot')
    plt.grid(True)
    plt.show()

# Create the elbow plot for the selected features
plot_elbow(scaled_features)


# # CORNER PLOT

# In[49]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
file_path = 'C:/Users/91910/OneDrive/Desktop/assignment_data/ICRISAT.csv'
data = pd.read_csv(file_path)

# Select numerical columns for clustering and visualization
numerical_columns = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'WHEAT PRODUCTION (1000 tons)']
selected_data = data[numerical_columns].dropna()

# Scale the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_data)

# Perform K-Means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original dataset
selected_data['Cluster'] = cluster_labels

# Create a corner plot (pairplot) with clusters
sns.pairplot(
    selected_data,
    diag_kind='hist',
    hue='Cluster',
    palette='viridis',
    corner=True,
    plot_kws={'s': 50, 'alpha': 0.7}
)

plt.suptitle("Corner Plot with Clusters", y=1.02, fontsize=16)
plt.show()


# In[ ]:




