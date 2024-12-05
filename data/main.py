import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import argparse
import numpy as np

MEASUREMENT_HEADER = ["ID", "Diag", "RTT", "T1", "T2", "T3", "T4", "RSSI", "RTT_raw", "RTT_est", "Dist_est"]
SPEED_OF_LIGHT_METERS_PER_SECOND: float = 299_792_458 # m/s
ROTATION_TEST_TRUE_DISTANCE_CM = 500
ROTATION_FRONTAL_ANGLE_DEGREES = 277 # in degrees; depends on the used angle for the experiment
path_to_data = ""

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

    if plot_type == "d":
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
    plt.scatter(data['Dist_true_cm'], data['Dist_difference'], color = 'blue', alpha=0.1)
    plt.xlabel('Real distance (cm)')
    plt.ylabel('Difference to real distance (cm)')

    plt.scatter(data['Dist_true_cm'], data['Dist_average'], color='red')
    plt.scatter(data['Dist_true_cm'], data['Dist_median'], color='yellow')

    # x and y scaling should be the same
    # plt.gca().set_aspect('equal', adjustable='box')

    # plot the optimal line as well in red
    # plt.plot([0, 3000], [0, 3000], 'r')

    # plot horizontal line at 0
    plt.axhline(0, color='black', lw=1)
    plt.legend(["Distances differences" , "Average distance differences", "Median distance differences"], ncol = 1 , loc = "upper left")

    plt.savefig(os.path.join(path_to_data, "graphs", "real_vs_measured.png"))
    # plt.show()
    plt.close()

def plot_individual_distance_measurement(data: pd.DataFrame):
    # plot a histogram
    plt.figure(figsize=(10,6))
    plt.title(str(data['Dist_true_cm'][0]) + " cm")
    plt.xlabel('Measured distance (cm)')
    plt.ylabel('Count in bucket')

    # increase bin size
    real_distance = data['Dist_true_cm'][0]
    minimum_distance = data['Dist_calculated'].min()
    maximum_distance = data['Dist_calculated'].max()
    width = maximum_distance - minimum_distance 
    plt.axvline(x=data['Dist_calculated'].mean(), color='red', linestyle='--')
    plt.axvline(x=data['Dist_calculated'].median(), color='yellow', linestyle=':')
    plt.axvline(x=real_distance, color='blue')
    plt.grid()
    

    plt.hist(data['Dist_calculated'], bins=range(minimum_distance, maximum_distance + 20, 20), color='lightblue', edgecolor='black')
    plt.savefig(os.path.join(path_to_data, "graphs", str(real_distance) + "cm"))
    plt.close()


def plot_ecdf_distance_measurement(df):
    real_distance = df['Dist_true_cm'][0]
    plt.ecdf(x=df['Dist_calculated'])

    plt.title("ECDF Graph for " + str(real_distance) + " cm")
    plt.axvline(real_distance, color="red", linestyle="-")

    plt.xlabel("Distance in cm")
    plt.ylabel("Distribution of values")

    plt.savefig(os.path.join(path_to_data, "graphs", "ecdf_" + str(real_distance) + "cm"))
    plt.close()

def plot_time_graph(df: pd.DataFrame):
    real_distance = df['Dist_true_cm'][0]
    plt.scatter(df["T1"], df["Dist_calculated"], alpha=0.3)
    plt.title("Time diagram for " + str(real_distance) + " cm")
    plt.xlabel("Time of the measurement (T1) [picoseconds]")
    plt.ylabel("Distance [cm]")
    plt.savefig(os.path.join(path_to_data, "graphs", "time_" + str(real_distance) + "cm"))
    plt.close()

def calculate_distance_with_rtt(rtt: int) -> int:
    return int(rtt * SPEED_OF_LIGHT_METERS_PER_SECOND * 100 / 1e+12 / 2)

def add_distance_with_rtt(df: pd.DataFrame):
    df['Dist_calculated'] = df['RTT'].apply(func=calculate_distance_with_rtt)
    df['Dist_difference'] = df['Dist_calculated'] - df['Dist_true_cm'] 

def average_distance_difference(data: pd.DataFrame):
    # group by real distance
    # calculate the mean of the difference
    averages_df = data.groupby('Dist_true_cm')['Dist_difference'].mean()
    medians_df = data.groupby('Dist_true_cm')['Dist_difference'].median()
    # expand the df again, so that the average distance is also a column
    data['Dist_average'] = data['Dist_true_cm'].map(averages_df)
    data['Dist_median'] = data['Dist_true_cm'].map(medians_df)

def plot_radial_graph(df: pd.DataFrame):
    real_distance = df['Dist_true_cm'][0]
    plt.title("Sensor attached to the saddle - top down view - bicycle faces north (0°)")
    plt.figure()
    reduced_df = df.drop_duplicates('Angle')

    # convert to radians
    reduced_df['Angle'] = np.radians(reduced_df['Angle'])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(reduced_df['Angle'], reduced_df['Dist_average'], c='blue', label="Data Points")
    plt.savefig(os.path.join(path_to_data, "graphs", "radial_" + str(real_distance) + "cm"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("plot_type", help="Choose between distance plots (d) or antenna plot (r)")
    parser.add_argument("dirname", help="Directory name for the data files")

    args = parser.parse_args()

    global path_to_data

    if args.plot_type == "d":
            path_to_data = os.path.join(path_to_data, "different_distances")
    elif args.plot_type == "r":
            path_to_data = os.path.join(path_to_data, "different_angles_saddle")

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
        plot_radial_graph(df)

if __name__ == '__main__':
    main()