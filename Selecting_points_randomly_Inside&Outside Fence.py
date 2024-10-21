import os
import time
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Log function to write messages to a file
def log_timing(message, log_file='timing_log.txt'):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# Function to load a single GeoLife trajectory file
def load_geolife_trajectory(file_path):
    columns = ['latitude', 'longitude', 'zero', 'altitude', 'date_days', 'date', 'time']
    data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y-%m-%d %H:%M:%S')
    return data[['latitude', 'longitude', 'altitude', 'timestamp']]

# Load all trajectory files in the dataset folder
def load_geolife_dataset(folder_path):
    trajectory_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.plt'):
                file_path = os.path.join(root, file)
                trajectory_data.append(load_geolife_trajectory(file_path))
    return pd.concat(trajectory_data, ignore_index=True)

# Haversine formula to calculate distance between two points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Function to calculate distances for all points once
def calculate_distances(data, center_lat, center_lon):
    data['distance'] = data.apply(lambda row: haversine(center_lon, center_lat, row['longitude'], row['latitude']), axis=1)
    return data

# Function to filter and randomly sample coordinates inside the geofence
def sample_coordinates_within_geofence(data, center_lat, center_lon, radius, sample_size=50):
    start_time = time.perf_counter()
    
    # Filter data for points inside the geofence
    inside_geofence = data[data['distance'] <= radius]
    
    # Check if there are enough coordinates inside the geofence
    num_available = len(inside_geofence)
    if num_available < sample_size:
        print(f"Only {num_available} coordinates available inside the geofence. Sampling all available points.")
        sample_size = num_available  # Adjust sample_size to the number of available points
    
    # Randomly sample the coordinates
    sampled_data = inside_geofence.sample(n=sample_size, random_state=1)  # random_state for reproducibility
    
    end_time = time.perf_counter()
    
    # Calculate and print processing time
    processing_time = end_time - start_time
    log_timing(f"Time taken to sample {sample_size} coordinates inside geofence: {processing_time:.4f} seconds", log_file)
    print(f"Time taken to sample {sample_size} coordinates inside geofence: {processing_time:.4f} seconds")
    #return sampled_data[['latitude', 'longitude', 'timestamp']]  # Return relevant columns

# Function to filter and randomly sample coordinates outside the geofence
def sample_coordinates_outside_geofence(data, center_lat, center_lon, radius, sample_size=50):
    start_time = time.perf_counter()
    
    # Filter data for points outside the geofence
    outside_geofence = data[data['distance'] > radius]
    
    # Check if there are enough coordinates outside the geofence
    num_available = len(outside_geofence)
    if num_available < sample_size:
        print(f"Only {num_available} coordinates available outside the geofence. Sampling all available points.")
        sample_size = num_available  # Adjust sample_size to the number of available points
    
    # Randomly sample the coordinates
    sampled_data = outside_geofence.sample(n=sample_size, random_state=1)  # random_state for reproducibility
    
    end_time = time.perf_counter()
    
    # Calculate and print processing time
    processing_time = end_time - start_time
    log_timing(f"Time taken to sample {sample_size} coordinates outside geofence: {processing_time:.4f} seconds", log_file)

    print(f"Time taken to sample {sample_size} coordinates outside geofence: {processing_time:.4f} seconds")
    
    #return sampled_data[['latitude', 'longitude', 'timestamp']]  # Return relevant columns

# Perturbation function for Local Differential Privacy (LDP)
def perturb_geospatial_data(data, epsilon):
    noise = np.random.laplace(loc=0.0, scale=1.0/epsilon, size=data.shape)
    perturbed_data = data + noise
    # Ensure latitude and longitude stay within valid ranges after perturbation
    perturbed_data[:, 0] = np.clip(perturbed_data[:, 0], -90.0, 90.0)  # Latitude
    perturbed_data[:, 1] = np.clip(perturbed_data[:, 1], -180.0, 180.0)  # Longitude
    return perturbed_data

# Function to filter and randomly sample perturbed coordinates inside the geofence
def sample_perturbed_coordinates_within_geofence(data, center_lat, center_lon, radius, epsilon, sample_size=50):
    start_time = time.perf_counter()
    
    # Filter data for points inside the geofence
    inside_geofence = data[data['distance'] <= radius]
    
    # Check if there are enough coordinates inside the geofence
    num_available = len(inside_geofence)
    if num_available < sample_size:
        print(f"Only {num_available} coordinates available inside the geofence. Sampling all available points.")
        sample_size = num_available  # Adjust sample_size to the number of available points
    
    # Randomly sample the coordinates
    if num_available > 0:
        sampled_data = inside_geofence.sample(n=sample_size, random_state=1)  # random_state for reproducibility
        
        # Perturb the coordinates using the LDP mechanism
        perturbed_data = perturb_geospatial_data(sampled_data[['latitude', 'longitude']].values, epsilon)
        perturbed_data_df = pd.DataFrame(perturbed_data, columns=['latitude', 'longitude'])
        perturbed_data_df['timestamp'] = sampled_data['timestamp'].values  # Add the original timestamps
        perturbed_data_df['distance'] = sampled_data['distance'].values  # Add the distance
    
    else:
        print("No coordinates available inside the geofence.")
        perturbed_data_df = pd.DataFrame()  # Empty DataFrame if no points
    
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    log_timing(f"Time taken to sample {sample_size} perturbed coordinates inside geofence: {processing_time:.4f} seconds", log_file)

    print(f"Time taken to sample {sample_size} perturbed coordinates inside geofence: {processing_time:.4f} seconds")
    
    # Print and return all perturbed coordinates
    #print(perturbed_data_df)
    #return perturbed_data_df if not perturbed_data_df.empty else None

# Function to filter and randomly sample perturbed coordinates outside the geofence
def sample_perturbed_coordinates_outside_geofence(data, center_lat, center_lon, radius, epsilon, sample_size=50):
    start_time = time.perf_counter()

    # Perturb the latitude and longitude columns with LDP
    perturbed_data = perturb_geospatial_data(data[['latitude', 'longitude']].values, epsilon)
    
    # Replace the original latitude and longitude with perturbed values in the DataFrame
    data[['latitude', 'longitude']] = perturbed_data

    # Recalculate distances based on the perturbed data
    data = calculate_distances(data, center_lat, center_lon)

    # Filter data for points outside the geofence
    outside_geofence = data[data['distance'] > radius]

    # Check if there are enough coordinates outside the geofence
    num_available = len(outside_geofence)
    if num_available < sample_size:
        print(f"Only {num_available} coordinates available outside the geofence. Sampling all available points.")
        sample_size = num_available  # Adjust sample_size to the number of available points

    # Randomly sample the coordinates
    sampled_data = outside_geofence.sample(n=sample_size, random_state=1)  # random_state for reproducibility

    end_time = time.perf_counter()

    # Calculate and print processing time
    processing_time = end_time - start_time
    log_timing(f"Time taken to sample {sample_size} perturbed coordinates outside geofence: {processing_time:.4f} seconds", log_file)
    print(f"Time taken to sample {sample_size} perturbed coordinates outside geofence: {processing_time:.4f} seconds")
    #return sampled_data[['latitude', 'longitude', 'timestamp']]  # Return relevant columns


# Example usage
geolife_folder = 'C:\\Users\\user\\Downloads\\Geofencing-with-Weather-API\\Geofencing-with-Weather-API\\Geolife Trajectories 1.3'
if os.path.exists(geolife_folder):
    trajectory_data = load_geolife_dataset(geolife_folder)
else:
    raise Exception("Dataset folder path does not exist!")

center_lat = 39.9087  # Beijing center latitude
center_lon = 116.3975  # Beijing center longitude
radius = 1.0  # Radius in kilometers
epsilon = 1.0  # Privacy parameter for LDP

# Calculate distances for all points once
trajectory_data = calculate_distances(trajectory_data, center_lat, center_lon)

# Log file path
log_file = 'geofence_timing_log20.txt'

# Sample 50 coordinates inside the geofence
sampled_coordinates_inside = sample_coordinates_within_geofence(trajectory_data, center_lat, center_lon, radius)
print(sampled_coordinates_inside)

# Sample 50 coordinates outside the geofence
sampled_coordinates_outside = sample_coordinates_outside_geofence(trajectory_data, center_lat, center_lon, radius)
print(sampled_coordinates_outside)

# Sample 50 perturbed coordinates inside the geofence
sampled_perturbed_coordinates_inside = sample_perturbed_coordinates_within_geofence(trajectory_data, center_lat, center_lon, radius, epsilon)
print(sampled_perturbed_coordinates_inside)

# Sample 50 perturbed coordinates outside the geofence
sampled_perturbed_coordinates_outside = sample_perturbed_coordinates_outside_geofence(trajectory_data, center_lat, center_lon, radius, epsilon)
print(sampled_perturbed_coordinates_outside)

