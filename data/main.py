import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from pyubx2 import UBXReader

import argparse
import numpy as np
from PIL import Image
import geopy.distance

MEASUREMENT_HEADER = ["ID", "Diag", "RTT", "T1", "T2", "T3", "T4", "RSSI", "RTT_raw", "RTT_est", "Dist_est"]
SPEED_OF_LIGHT_METERS_PER_SECOND: float = 299_792_458 # m/s
ROTATION_TEST_TRUE_DISTANCE_CM = 500
ROTATION_FRONTAL_ANGLE_DEGREES = 277 # in degrees; depends on the used angle for the experiment
BICYCLE_IMAGE_NAME = "bicycle.png"
path_to_data = ""

# Open an image from a computer 
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output

def get_data_file_names(ending: str) -> list:
    return [f for f in os.listdir(path_to_data) if f.endswith(ending)]

def clean_input(input_data: str) -> str:
    clean_output = []
    # uses regex to check if the line starts with a number and a comma
    for line in input_data.split('\n'):
        if not re.match(r'^\d+,', line):
            continue
        clean_output.append(line)

    return '\n'.join(clean_output)

def load_file_content(file_path: str) -> str:
    with open(os.path.join(path_to_data, file_path), 'r') as file:
        return file.read()

# loads the file and removes all unnecessary lines that don't containt data
def load_and_clean_file(file_path: str) -> pd.DataFrame:
    file_content = load_file_content(file_path)
    return clean_input(file_content)

def convert_to_dataframe(file_content: str):
    # splits the file content into lines
    lines = file_content.split('\n')
    # creates a list of lists
    data = [[int(item) for item in line.split(',')] for line in lines if line.strip()]
    # creates a dataframe with the data and the header
    # the data items are all integers
    return pd.DataFrame(data, columns=MEASUREMENT_HEADER)
    
def check_data(data: pd.DataFrame):
    # a few things most hold true for the data
    # the RTT value must be greater than 0
    # the RTT value must be equal to the difference between T4-T1 and T3-T2
    assert (data["RTT"] > 0).all()
    assert (data["RTT"] ==  (data["T4"] - data["T1"]) - (data["T3"] - data["T2"])).all()


def load_multiple_files(file_names: list, plot_type: str):
    file_contents = [load_and_clean_file(file_name) for file_name in file_names]
    dfs = [convert_to_dataframe(file_content) for file_content in file_contents]

    if plot_type == "d" or plot_type == "dg":
        for df, name in zip(dfs, file_names):
            df['Dist_true_cm'] = int(''.join(name[0:2]))*100
            check_data(df)
    
    # this adds the angle instead of the true distance
    elif plot_type == "r":
        for df, name in zip(dfs, file_names):
            normalized_angle = int(''.join(name[0:3])) - ROTATION_FRONTAL_ANGLE_DEGREES
            if normalized_angle < 0:
                normalized_angle += 360

            df['Angle'] = normalized_angle
            df['Dist_true_cm'] = ROTATION_TEST_TRUE_DISTANCE_CM
            check_data(df)

    
    combined_df = pd.concat(dfs)
    return combined_df


def plot_real_distance_vs_estimated_distance(data: pd.DataFrame):
    # plots the real distance vs the estimated distance
    plt.scatter(data['Dist_true_cm']/100, data['Dist_difference']/100, color = 'blue', alpha=0.1)
    plt.xlabel('Real distance (m)')
    plt.ylabel('Difference to real distance (m)')

    plt.scatter(data['Dist_true_cm']/100, data['Dist_difference_average']/100, color='red')
    plt.scatter(data['Dist_true_cm']/100, data['Dist_difference_median']/100, color='yellow')

    # x and y scaling should be the same
    # plt.gca().set_aspect('equal', adjustable='box')

    # plot the optimal line as well in red
    # plt.plot([0, 3000], [0, 3000], 'r')

    # plot horizontal line at 0
    plt.axhline(0, color='black', lw=1)
    plt.legend(["Distances differences" , "Average distance differences", "Median distance differences"], ncol = 1 , loc = "upper left")

    plt.savefig(os.path.join(path_to_data, "graphs", "real_vs_measured.pdf"))
    # plt.show()
    plt.close()

def plot_individual_distance_measurement(data: pd.DataFrame):
    # plot a histogram
    plt.figure(figsize=(10,6))
    real_distance = data['Dist_true_cm'][0]//100

    plt.title(str(real_distance) + " m")
    plt.xlabel('Measured distance (m)')
    plt.ylabel('Count in bucket')

    # increase bin size

    minimum_distance = data['Dist_calculated'].min()//100
    maximum_distance = data['Dist_calculated'].max()//100
    width = maximum_distance - minimum_distance 
    plt.axvline(x=data['Dist_calculated'].mean()/100, color='red', linestyle='--')
    plt.axvline(x=data['Dist_calculated'].median()/100, color='yellow', linestyle=':')
    plt.axvline(x=real_distance, color='blue')
    plt.grid()
    

    plt.hist(data['Dist_calculated']/100, bins=range(minimum_distance, maximum_distance + 1, 1), color='lightblue', edgecolor='black')
    plt.savefig(os.path.join(path_to_data, "graphs", str(real_distance) + "m.pdf"))
    plt.close()

def plot_ecdf_distance_measurement(df):
    real_distance = df['Dist_true_cm'][0]//100
    plt.ecdf(x=df['Dist_calculated']/100)

    plt.title("ECDF Graph for " + str(real_distance) + " m")
    plt.axvline(real_distance, color="red", linestyle="-")

    plt.xlabel("Distance [m]")
    plt.ylabel("Distribution of values")

    plt.savefig(os.path.join(path_to_data, "graphs", "ecdf_" + str(real_distance) + "m.pdf"))
    plt.close()

def plot_time_graph(df: pd.DataFrame):
    real_distance = df['Dist_true_cm'][0]//100
    # convert T1 in picoseconds to seconds and add a column to the dataframe as timestamp
    df['T1_s'] = df['T1'] / 1e+12

    plt.scatter(df["T1_s"], df["Dist_calculated"]/100, alpha=0.3)
    plt.title("Time diagram for " + str(real_distance) + " m")
    plt.xlabel("Time of the measurement (T1) [seconds]")
    plt.ylabel("Distance [m]")
    plt.savefig(os.path.join(path_to_data, "graphs", "time_" + str(real_distance) + "m.pdf"))
    plt.close()

def calculate_distance_with_rtt(rtt: int) -> int:
    return int(rtt * SPEED_OF_LIGHT_METERS_PER_SECOND * 100 / 1e+12 / 2)

def add_distance_with_rtt(df: pd.DataFrame):
    df['Dist_calculated'] = df['RTT'].apply(func=calculate_distance_with_rtt)
    df['Dist_difference'] = df['Dist_calculated'] - df['Dist_true_cm'] 

def average_distance_difference(data: pd.DataFrame):
    # group by real distance
    # calculate the mean of the difference
    average_difference_df = data.groupby('Dist_true_cm')['Dist_difference'].mean()
    median_difference_df = data.groupby('Dist_true_cm')['Dist_difference'].median()
    # expand the df again, so that the average distance is also a column
    data['Dist_difference_average'] = data['Dist_true_cm'].map(average_difference_df)
    data['Dist_difference_median'] = data['Dist_true_cm'].map(median_difference_df)


    if 'Angle' in data.columns:
        average_df = data.groupby('Angle')['Dist_calculated'].mean()
        median_df = data.groupby('Angle')['Dist_calculated'].median()
        data['Dist_average'] = data['Angle'].map(average_df)
        data['Dist_median'] = data['Angle'].map(median_df)

def rssi_calculations(df: pd.DataFrame):
    # group by angle and calculate the average RSSI
    average_rssi_df = df.groupby('Angle')['RSSI'].mean()
    median_rssi_df = df.groupby('Angle')['RSSI'].median()

    # expand the df again, so that the average/mean RSSI is also a column
    df['RSSI_average'] = df['Angle'].map(average_rssi_df)
    df['RSSI_median'] = df['Angle'].map(median_rssi_df)

def plot_radial_graph_with_distances(df: pd.DataFrame):
    real_distance: int = df['Dist_true_cm'].iloc[0]
    # reduced_df = df.drop_duplicates('Angle').copy()

    print(df[df['Angle'] == 270]['RSSI'])

    # convert to radians
    df.loc[:, 'Angle_radians'] = np.radians(df['Angle'])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})


    # increase the size of the plot
    fig.set_size_inches(10, 10)
    
    # Achseneinteilung auf 0 in die Mitte
    # limit the radius
    ax.set_ylim(0, 1000)

    # draw a circle with the real distance
    angles = np.linspace(0, 2*np.pi, 100)
    radii = np.full(100, real_distance)

    ax.plot(angles, radii, color='green', linestyle='--')

    # make 0 degrees at the top
    ax.set_theta_offset(np.pi/2)    

    ax.scatter(df['Angle_radians'], df['Dist_calculated'], c='blue', label="Measured distance", alpha=0.1)
    ax.scatter(df['Angle_radians'], df['Dist_average'], c='red', label="Average")
    ax.scatter(df['Angle_radians'], df['Dist_median'], c='yellow', label="Median")

    # add meter suffix to the radius
    # ax.set_yticklabels([str(i) + " m" for i in range(0, 11, 1)])

    plt.legend(["Real Distance", "Distances" , "Average distance", "Median distance"], ncol = 1 , loc = "upper left")
    plt.title("Measured distances for " + str(real_distance/100) + " m")

    # the image should be in the middle of the radial plot
    image_xaxis = 0.465
    image_yaxis = 0.465
    image = open_image_local(BICYCLE_IMAGE_NAME)

    # rotate the image
    image = np.rot90(image, k=1)

    ax_image = fig.add_axes([image_xaxis, image_yaxis, 0.1, 0.1])
    ax_image.imshow(image)
    ax_image.axis('off') 

    fig.savefig(os.path.join(path_to_data, "graphs", "radial_" + str(real_distance) + "cm.pdf"))
    plt.close()


def plot_radial_graph_with_rssi(df: pd.DataFrame):
    real_distance: int = df['Dist_true_cm'].iloc[0]


    # convert to radians
    df.loc[:, 'Angle_radians'] = np.radians(df['Angle'])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # label the radius as RSSI
    ax.set_ylabel('RSSI')
     

    # increase the size of the plot
    fig.set_size_inches(10, 10)

    # make 0 degrees at the top
    ax.set_theta_offset(np.pi/2)


    ax.scatter(df['Angle_radians'], df['RSSI'], c='blue', label="Measured RSSI", alpha=0.1)
    ax.scatter(df['Angle_radians'], df['RSSI_average'], c='red', label="Average RSSI")
    ax.scatter(df['Angle_radians'], df['RSSI_median'], c='yellow', label="Median RSSI")

    # plot a line that goes through all median values
    # ax.plot(df['Angle_radians'], df['RSSI_median'], c='yellow', linestyle='--')

    plt.legend(["RSSI" , "Average RSSI", "Median RSSI"], ncol = 1 , loc = "upper left")
    plt.title("RSSI values for " + str(real_distance//100) + " m")

    # the image should be in the middle of the radial plot
    image_xaxis = 0.465
    image_yaxis = 0.465
    image = open_image_local(BICYCLE_IMAGE_NAME)

    # rotate the image
    image = np.rot90(image, k=1)

    ax_image = fig.add_axes([image_xaxis, image_yaxis, 0.1, 0.1])
    ax_image.imshow(image)
    ax_image.axis('off') 

    fig.savefig(os.path.join(path_to_data, "graphs", "radial_rssi.pdf"))
    plt.close()

"""
def load_gps_data(file_path: str, df: pd.DataFrame) -> pd.DataFrame:
    # the gps data is a csv file with two columns
    # first column is the real distance in meters, second column is the measured distance in meters
    gps_data = pd.read_csv(file_path, names=['Dist_true_cm', 'GPS_distance'])
    gps_data['GPS_distance_difference'] = gps_data['Dist_true_cm'] - gps_data['GPS_distance']
    # merge the gps data with the existing data
    df = pd.merge(df, gps_data, on='Dist_true_cm', how='left')
    print(df.columns)
    print(df.head())
    return df
"""

# this only looks at GNGGA messages
def convert_ubx_to_data(dir) -> pd.DataFrame:
    # UBX files are basically CSV already
    #GNGGA,['160345.00', '5101.74467', 'N', '01345.09246', 'E', '2', '12', '0.78', '134.6', 'M', '43.7', 'M', '', '0000']
    # Define the column names and data types
    column_names = ['Real_Distance_m', 'Time', 'Latitude', 'N/S', 'Longitude', 'E/W', 'Quality', 'Satellites', 'HDOP', 'Altitude', 'AltitudeVal', 'Geosep', 'GeosepVal', 'Age', 'Station']
    data_types = {
        'Real_Distance_m': int,
        'Time': str,
        'Latitude': str,
        'N/S': str,
        'Longitude': str,
        'E/W': str,
        'Quality': int,
        'Satellites': int,
        'HDOP': float,
        'Altitude': str,
        'AltitudeVal': str,
        'Geosep': str,
        'GeosepVal': str,
        'Age': str,
        'Station': str
    }
    dfs = []

    for file in os.listdir(dir):
        if not file.endswith(".ubx"):
            continue
        real_distance: str = file.split('-')[-1].split('.')[0]

        df = pd.DataFrame(columns=column_names)

        # first we need to filter for GNGAA messages
        with open(os.path.join(dir, file), 'r', errors='replace') as f:
            for line in f:
                if not line.startswith("$GNGGA"):
                    continue

                            # Split the line into columns and insert the real distance
                row = [real_distance] + line.strip().split(',')[1:]
                # add row to pandas dataframe
                df.loc[len(df)] = row

            df = df.astype(data_types)
            
            # remove all rows where lat/long is empty
            df = df[df['Latitude'] != '']
            df = df[df['Longitude'] != '']

            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def convert_dms_to_dd(df: pd.DataFrame) -> float:
    # convert the latitude and longitude from degrees, minutes, seconds to decimal degrees
    # the latitude and longitude are in the format ddmm.mmmm

    # convert the latitude
    df['Latitude'] = df['Latitude'].apply(lambda x: int(x[:2]) + float(x[2:])/60)
    # convert the longitude
    df['Longitude'] = df['Longitude'].apply(lambda x: int(x[:3]) + float(x[3:])/60)

mapping_of_dir_to_start_pos = {
    "02": (51.029150, 13.751346),
    "03": (51.02996886808531, 13.74926083421233),
}

def calculate_distance_with_gps(gps_data: pd.DataFrame) -> pd.DataFrame:
    # take the average latitude and longitude of Real_Distance_m == 0
    # and calculate the distance to the average latitude and longitude of the other distances
    # the distance is in meters
    position = mapping_of_dir_to_start_pos[path_to_data.split('/')[-1]]

    print("The start latitude is: " + str(position[0]))
    print("The start longitude is: " + str(position[1]))

    # calculate the distance between the start and the average latitude and longitude
    gps_data['Dist_calculated'] = gps_data.apply(lambda x: geopy.distance.distance(position, (x['Average_Latitude'], x['Average_Longitude'])).m, axis=1)

    # now calculate the difference between the real distance and the calculated distance
    gps_data['Dist_difference'] = gps_data['Dist_calculated'] - gps_data['Real_Distance_m']

    # remove all rows where the real distance is 0
    gps_data.drop(gps_data[gps_data['Real_Distance_m'] == 0].index, inplace=True)

def calculate_average_lat_lon(gps_data: pd.DataFrame):
    # group by the real distance
    # calculate the average latitude and longitude

    average_lat = gps_data.groupby('Real_Distance_m')['Latitude'].mean()
    average_lon = gps_data.groupby('Real_Distance_m')['Longitude'].mean()

    gps_data['Average_Latitude'] = gps_data['Real_Distance_m'].map(average_lat)
    gps_data['Average_Longitude'] = gps_data['Real_Distance_m'].map(average_lon)

def plot_distances_with_gps(ftm_data: pd.DataFrame, gps_data: pd.DataFrame):
    # first print the distances in the df as a scatter plot
    # plt.scatter(ftm_data['Dist_true_cm']/100, ftm_data['Dist_difference']/100, color = 'blue', alpha=0.1)
    plt.scatter(ftm_data['Dist_true_cm']/100, ftm_data['Dist_difference_average']/100, color='red')
    plt.xlabel('Real distance (m)')
    plt.ylabel('Difference to real distance (m)')

    # plt.scatter(df['Dist_true_cm']/100, df['Dist_difference_average'], color='red')
    plt.xlim(0, min(gps_data['Real_Distance_m'].max(), (ftm_data['Dist_true_cm']/100).max()) + 1)


    # print the distances in the gps data as a scatter plot
    plt.scatter(gps_data['Real_Distance_m'], gps_data['Dist_difference'], color='green')

    plt.legend(["Average FTM distance difference", "Average GPS distances differences"], ncol = 1 , loc = "upper left")

    # draw a horizontal line at 0
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.grid()
    plt.savefig(os.path.join(path_to_data, "graphs", "gps_distances.pdf"))
    plt.close()

def plot_number_of_satellites(gps_data: pd.DataFrame):
    # plot the average number of satellites
    plt.scatter(gps_data['Real_Distance_m'], gps_data['Satellites'], color='blue', alpha=0.1)
    plt.xlabel('Real distance (m)')
    plt.ylabel('Number of satellites')
    plt.grid()
    plt.savefig(os.path.join(path_to_data, "graphs", "satellites.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("plot_type", help="Choose between distance plots (d), antenna plot (r), or distance with gps (dg)")
    parser.add_argument("dirname", help="Directory name for the data files")

    args = parser.parse_args()

    global path_to_data

    if args.plot_type == "d":
            path_to_data = os.path.join(path_to_data, "different_distances")
    elif args.plot_type == "r":
            path_to_data = os.path.join(path_to_data, "different_angles_saddle")
    elif args.plot_type == "dg":
            path_to_data = os.path.join(path_to_data, "different_distances")
    else:
        print("Invalid plot type")
        return


    path_to_data = os.path.join(path_to_data, args.dirname)


    df = load_multiple_files(get_data_file_names('.out'), args.plot_type)
    add_distance_with_rtt(df)
    average_distance_difference(df)

    if args.plot_type == "d":

        # prints the overall graph that compares distances
        plot_real_distance_vs_estimated_distance(df)

        # prints the indivdiual distances as a histogram
        # go over each real distance
        for dist in df['Dist_true_cm'].unique():
            subset = df[df['Dist_true_cm'] == dist]
            # sort
            subset = subset.sort_values(by='Dist_true_cm')
            plot_individual_distance_measurement(subset)
            plot_ecdf_distance_measurement(subset)
            plot_time_graph(subset)
    elif args.plot_type == "r":
        rssi_calculations(df)
        plot_radial_graph_with_distances(df)
        plot_radial_graph_with_rssi(df)
    elif args.plot_type == "dg":
        gps_data = convert_ubx_to_data(os.path.join("gps", args.dirname))
        convert_dms_to_dd(gps_data)
        calculate_average_lat_lon(gps_data)
        calculate_distance_with_gps(gps_data)
        plot_distances_with_gps(df, gps_data)
        plot_number_of_satellites(gps_data)

if __name__ == '__main__':
    main()