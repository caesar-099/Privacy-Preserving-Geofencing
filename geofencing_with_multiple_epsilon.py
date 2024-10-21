import os
import pandas as pd
import logging
from math import radians, cos, sin, asin, sqrt
import numpy as np
import matplotlib.pyplot as plt

# Define the path for the log file
log_file_path = 'C:\\Users\\user\\Desktop\\geofencing\\accuracytesting\\geofencing_accuracydata_with_multiple_epsilon_10_r.txt'

# Create the directory if it doesn't exist
log_dir = os.path.dirname(log_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load all trajectory files in the dataset folder
def load_geolife_dataset(folder_path):
    trajectory_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.plt'):
                file_path = os.path.join(root, file)
                trajectory_data.append(load_geolife_trajectory(file_path))
    return pd.concat(trajectory_data, ignore_index=True)

# Function to load a single GeoLife trajectory file
def load_geolife_trajectory(file_path):
    columns = ['latitude', 'longitude', 'zero', 'altitude', 'date_days', 'date', 'time']
    data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y-%m-%d %H:%M:%S')
    return data[['latitude', 'longitude', 'altitude', 'timestamp']]

# Perturbation function for Local Differential Privacy (LDP)
def perturb_geospatial_data(data, epsilon):
    noise = np.random.laplace(loc=0.0, scale=1.0/epsilon, size=data.shape)
    perturbed_data = data + noise
    # Ensure latitude and longitude stay within valid ranges after perturbation
    perturbed_data[:, 0] = np.clip(perturbed_data[:, 0], -90.0, 90.0)  # Latitude
    perturbed_data[:, 1] = np.clip(perturbed_data[:, 1], -180.0, 180.0)  # Longitude
    return perturbed_data

# Haversine formula to calculate distance between two points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Geofencing check function
def check_geofence_status(data, center_lat, center_lon, radius):
    status_list = []
    for index, row in data.iterrows():
        lat2 = row['latitude']
        lon2 = row['longitude']
        distance = haversine(center_lon, center_lat, lon2, lat2)

        status = 'Outside Geofence'
        if distance <= radius:
            status = 'Inside Geofence'
        elif radius < distance <= radius + 0.2:
            status = 'Geofence within 200 metres'
        
        status_list.append(status)
    return status_list

# Accuracy comparison function
def compare_geofence_status(original_status, perturbed_status):
    correct_count = sum([1 for orig, pert in zip(original_status, perturbed_status) if orig == pert])
    accuracy = correct_count / len(original_status)
    return accuracy

# Accuracy comparison process
def evaluate_geofence_accuracy(trajectory_data, center_lat, center_lon, radius, epsilon):
    # 1. Compute geofence status before perturbation (Original Data)
    original_status = check_geofence_status(trajectory_data, center_lat, center_lon, radius)
    original_accuracy = compare_geofence_status(original_status, original_status)  # Self-comparison for 100% accuracy
    logging.info(f"Epsilon: {epsilon}")
    logging.info(f"Radius: {radius}")
    logging.info(f"Accuracy before perturbation: {original_accuracy * 100:.2f}%")

    # 2. Perturb the trajectory data
    perturbed_coords = perturb_geospatial_data(trajectory_data[['latitude', 'longitude']].values, epsilon)
    perturbed_data = pd.DataFrame(perturbed_coords, columns=['latitude', 'longitude'])
    perturbed_data['timestamp'] = trajectory_data['timestamp']

    # 3. Compute geofence status after perturbation
    perturbed_status = check_geofence_status(perturbed_data, center_lat, center_lon, radius)

    # 4. Compare the geofence status for original vs perturbed data
    perturbed_accuracy = compare_geofence_status(original_status, perturbed_status)
    logging.info(f"Accuracy after perturbation: {perturbed_accuracy * 100:.2f}%")

    return original_accuracy, perturbed_accuracy

# Load dataset
geolife_folder = 'C:\\Users\\user\\Downloads\\Geofencing-with-Weather-API\\Geofencing-with-Weather-API\\Geolife Trajectories 1.3'
if os.path.exists(geolife_folder):
    trajectory_data = load_geolife_dataset(geolife_folder)
else:
    raise Exception("Dataset folder path does not exist!")

# Set parameters
center_lat = 39.9087  # Beijing center latitude
center_lon = 116.3975  # Beijing center longitude
radius = 0.1  # Radius in kilometers

# List of epsilon values for testing
epsilon_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
original_accuracies = []
perturbed_accuracies = []

# Evaluate accuracy for each epsilon value
for epsilon in epsilon_values:
    original_accuracy, perturbed_accuracy = evaluate_geofence_accuracy(trajectory_data, center_lat, center_lon, radius, epsilon)
    original_accuracies.append(original_accuracy)
    perturbed_accuracies.append(perturbed_accuracy)

# Plot the accuracies against epsilon values
plt.figure(figsize=(8, 6))
plt.plot(epsilon_values, [acc * 100 for acc in original_accuracies], marker='o', label='Original Accuracy')
plt.plot(epsilon_values, [acc * 100 for acc in perturbed_accuracies], marker='o', linestyle='--', label='Perturbed Accuracy')

# Add labels, title, and legend
plt.xlabel('Epsilon Value (Privacy Parameter)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Geofencing Before and After Perturbation')
plt.legend()
plt.grid(True)

# Define the path for saving the graph
graph_file_path = 'C:\\Users\\user\\Desktop\\geofencing\\accuracytesting\\geofencing_accuracy_comparison_10_r.png'

# Save the plot to a file
plt.savefig(graph_file_path)
# Show the plot
plt.show()
