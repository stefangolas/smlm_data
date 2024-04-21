# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:37:35 2024

@author: stefa
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from collections import Counter



file_path = "test_data.csv"

# Read CSV file into a DataFrame without header
df = pd.read_csv(file_path, header=None)

# Convert all elements to floats
df = df.apply(pd.to_numeric, errors='coerce')

# Extract x and y coordinates
x = df.iloc[:, 0]
y = df.iloc[:, 1]
complex_info = df.iloc[:, 4]

# Group data by complex
groups = df.groupby(complex_info)

# Create lists to store transformed x and y coordinates
transformed_x_list = []
transformed_y_list = []
previous_transforms = []

dist = 100
disp = 800

# Loop through groups, apply random xy transform, and store transformed coordinates
for name, group in groups:
    # Generate random x and y transforms
    random_x_transform = np.random.uniform(-disp, disp)
    random_y_transform = np.random.uniform(-disp, disp)
    
    # Adjust random transforms to ensure they are not within dist distance of any previous transform
    while any(abs(random_x_transform - tx) < dist and abs(random_y_transform - ty) < dist for tx, ty in previous_transforms):
        random_x_transform = np.random.uniform(-disp, disp)
        random_y_transform = np.random.uniform(-disp, disp)
    
    # Apply random transforms to group coordinates
    transformed_x = group.iloc[:, 0] + random_x_transform
    transformed_y = group.iloc[:, 1] + random_y_transform
    
    # Store the current transform to the list of previous transforms
    previous_transforms.append((random_x_transform, random_y_transform))
    
    # Store transformed coordinates
    transformed_x_list.append(transformed_x)
    transformed_y_list.append(transformed_y)

# Concatenate transformed coordinates lists into new columns
df['Transformed_X'] = pd.concat(transformed_x_list)
df['Transformed_Y'] = pd.concat(transformed_y_list)
print(df)
# Perform DBSCAN on transformed coordinates
transformed_data = df[['Transformed_X', 'Transformed_Y']]
dbscan = DBSCAN(eps=50, min_samples=5)
df['DBSCAN'] = dbscan.fit_predict(transformed_data)

# Plot data for each complex with random xy transform, colored by cluster membership
ax = plt.gca()  # Get current axis
ax.set_facecolor('black')  # Set background color to black

# Plot points colored by cluster membership
plt.scatter(df['Transformed_X'], df['Transformed_Y'], cmap='viridis', s=1/10)

#ax.set_xticks([])  # Remove x-axis ticks
#ax.set_yticks([])  # Remove y-axis ticks
ax.spines['bottom'].set_visible(False)  # Hide bottom spine
ax.spines['left'].set_visible(False)  # Hide left spine
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

plt.show()

# Function to create a BallTree from DataFrame
def create_ball_tree(df):
    points = df[['Transformed_X', 'Transformed_Y']].values
    return BallTree(points)

# Function to query BallTree for points within a given distance range
def find_points_within_distance_range(ball_tree, point, min_distance, max_distance):
    indices_within_max = ball_tree.query_radius([point], r=max_distance, return_distance=False)[0]
    indices_within_min = ball_tree.query_radius([point], r=min_distance, return_distance=False)[0]
    return list(set(indices_within_max) - set(indices_within_min))

# Function to calculate halfway point between two points
def calculate_halfway_point(point1, point2):
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

# Create a BallTree from DataFrame
ball_tree = create_ball_tree(df)

# Create a dictionary to store results
results = {}

# Iterate over rows and find points within distance range
for index, row in df.iterrows():
    point = [row['Transformed_X'], row['Transformed_Y']]
    points_within_distance_range = find_points_within_distance_range(ball_tree, point, min_distance=111, max_distance=113)  # Adjust distance range as needed
    halfway_points = [calculate_halfway_point(point, ball_tree.data[i]) for i in points_within_distance_range]
    results[index] = {
        'Transformed_X': row['Transformed_X'],
        'Transformed_Y': row['Transformed_Y'],
        'Points_within_distance_range': points_within_distance_range,
        'Halfway_Points': halfway_points
    }

# Extract halfway points from the results dictionary
halfway_points = []
for key, value in results.items():
    halfway_points.extend(value['Halfway_Points'])

# Convert halfway_points to a NumPy array for easier manipulation
halfway_points = np.array(halfway_points)

# Plot the halfway points
plt.scatter(halfway_points[:, 0], halfway_points[:, 1], color='red', label='Halfway Points',s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Halfway Points')
plt.legend()
plt.grid(True)
plt.show()


# Build a Ball Tree
ball_tree = BallTree(halfway_points)

# Define radius and minimum neighbors
radius = 60
min_neighbors = 20

# Initialize cluster labels
labels = np.full(len(halfway_points), -1)

# Assign clusters based on points within radius
current_label = 0
for i, point in enumerate(halfway_points):
    if labels[i] == -1:  # Check if point is not already assigned to a cluster
        # Query neighbors within the radius
        neighbors = ball_tree.query_radius([point], r=radius)[0]
        if len(neighbors) >= min_neighbors:  # Check if there are enough neighbors
            labels[neighbors] = current_label
            current_label += 1  # Move to the next cluster label

# Plot the halfway points colored by 'Cluster' labels
plt.scatter(halfway_points[:, 0], halfway_points[:, 1], c=labels, cmap='viridis',s=1)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Halfway Points Colored by Cluster Labels')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

# Determine majority cluster for each original point
for key, value in results.items():
    halfway_indices = value['Points_within_distance_range']
    halfway_labels = [labels[i] for i in halfway_indices]
    majority_label = Counter(halfway_labels).most_common(1)[0][0] if halfway_labels else -1  # Default label if no halfway points
    results[key]['Cluster'] = majority_label

# Plot the original points according to their assigned clusters
# Extract cluster labels
cluster_labels = [value['Cluster'] for value in results.values()]

# Get unique cluster labels
unique_clusters = np.unique(cluster_labels)

# Generate a color map
color_map = plt.cm.get_cmap('tab10', len(unique_clusters))

# Create a dictionary mapping cluster labels to colors
cluster_color_map = {cluster: color_map(i) for i, cluster in enumerate(unique_clusters)}

# Plot the transformed X and Y coordinates colored by 'Cluster'
for key, value in results.items():
    cluster_color = cluster_color_map.get(value['Cluster'], 'gray')  # Default to gray if cluster label not found in color map
    plt.scatter(value['Transformed_X'], value['Transformed_Y'], color=cluster_color, label='Cluster {}'.format(value['Cluster']), s=1)



# Match halfway points to their corresponding element in results and append labels
for halfway_point, label in zip(halfway_points, labels):
    halfway_point = np.array(halfway_point)  # Convert to NumPy array for comparison
    for key, value in results.items():
        for hp in value['Halfway_Points']:
            if np.array_equal(halfway_point, np.array(hp)):
                if 'labels' not in results[key]:
                    results[key]['labels'] = []
                results[key]['labels'].append(label)

# Calculate the most common label for each element and assign it to 'Cluster'
for key, value in results.items():
    if 'labels' in value:
        most_common_label = Counter(value['labels']).most_common(1)[0][0]
        results[key]['Cluster'] = most_common_label


from scipy.spatial import ConvexHull

# Extract Transformed_X, Transformed_Y, and Cluster for each element in results
x_values = []
y_values = []
clusters = []

# Filter out points belonging to the -1 cluster and collect points for each cluster
cluster_points = {}
for key, value in results.items():
    if value['Cluster'] != -1:
        cluster = value['Cluster']
        if cluster not in cluster_points:
            cluster_points[cluster] = {'x': [], 'y': []}
        cluster_points[cluster]['x'].append(value['Transformed_X'])
        cluster_points[cluster]['y'].append(value['Transformed_Y'])

# Plot the points colored by cluster membership
plt.figure(figsize=(8, 6))
for cluster, points in cluster_points.items():
    x_values = points['x']
    y_values = points['y']

    # Plot points
    plt.scatter(x_values, y_values, alpha=0.7,s=1)

    # Compute convex hull if there are more than 2 points in the cluster
    if len(x_values) > 2:
        points_array = np.column_stack((x_values, y_values))
        hull = ConvexHull(points_array)

        # Plot convex hull in white with thickness 5
        for simplex in hull.simplices:
            plt.plot(points_array[simplex, 0], points_array[simplex, 1], 'white', linewidth=3, alpha=0.5)

plt.scatter(df['Transformed_X'], df['Transformed_Y'], cmap='viridis', s=1/10)

#ax.set_xticks([])  # Remove x-axis ticks
#ax.set_yticks([])  # Remove y-axis ticks
ax.spines['bottom'].set_visible(False)  # Hide bottom spine
ax.spines['left'].set_visible(False)  # Hide left spine
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

plt.show()

# Extract Transformed_X, Transformed_Y, and DBSCAN cluster membership from the DataFrame
x_values = df['Transformed_X']
y_values = df['Transformed_Y']
dbscan_clusters = df['DBSCAN']

# Plot the points colored by DBSCAN cluster membership
plt.figure(figsize=(8, 6))
for cluster in np.unique(dbscan_clusters):
    if cluster != -1:  # Exclude noise points
        cluster_points = np.column_stack((x_values[dbscan_clusters == cluster], y_values[dbscan_clusters == cluster]))
        hull = ConvexHull(cluster_points)
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], '.', label=f'DBSCAN Cluster {cluster}')
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'white', linewidth=5, alpha=0.5)

#ax.set_xticks([])  # Remove x-axis ticks
#ax.set_yticks([])  # Remove y-axis ticks
ax.spines['bottom'].set_visible(False)  # Hide bottom spine
ax.spines['left'].set_visible(False)  # Hide left spine
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

plt.show()
