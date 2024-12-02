import pandas as pd
import matplotlib.pyplot as plt
import re
import os

MEASUREMENT_HEADER = ["ID", "Diag", "RTT", "T1", "T2", "T3", "T4", "RSSI", "RTT_raw", "RTT_est", "Dist_est"]
SPEED_OF_LIGHT_METERS_PER_SECOND: float = 299_792_458 # m/s

def get_data_file_names(ending: str) -> list:
    return [f for f in os.listdir() if f.endswith(ending)]

def clean_input(input_data: str) -> str:
    clean_output = []
    # uses regex to check if the line starts with a number and a comma
    for line in input_data.split('\n'):
        if not re.match(r'^\d+,', line):
            continue
        clean_output.append(line)

    return '\n'.join(clean_output)

def load_file_content(file_path: str) -> str:
    with open(file_path, 'r') as file:
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


def plot_data(data: pd.DataFrame):
    # plots
    pass

def load_multiple_files(file_names: list):
    file_contents = [load_and_clean_file(file_name) for file_name in file_names]
    dfs = [convert_to_dataframe(file_content) for file_content in file_contents]
    for df, name in zip(dfs, file_names):
        df['Dist_true_cm'] = int(''.join(name[0:2]))*100
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

    plt.savefig("real_vs_measured.png")
    # plt.show()

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
    plt.savefig(str(real_distance) + "cm")
    plt.cla()


def plot_ecdf_distance_measurement(subset):
    plt.cla()
    real_distance = subset['Dist_true_cm'][0]
    plt.ecdf(x=subset['Dist_calculated'])

    plt.savefig("ecdf_" + str(real_distance) + "cm")
    plt.cla()


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

def main():
    df = load_multiple_files(get_data_file_names('.out'))
    add_distance_with_rtt(df)

    # prints the overall graph that compares distances
    average_distance_difference(df)
    plot_real_distance_vs_estimated_distance(df)

    # prints the indivdiual distances as a histogram
    # go over each real distance
    for dist in df['Dist_true_cm'].unique():
        subset = df[df['Dist_true_cm'] == dist]
        # sort
        subset = subset.sort_values(by='Dist_true_cm')
        plot_individual_distance_measurement(subset)
        plot_ecdf_distance_measurement(subset)

if __name__ == '__main__':
    main()