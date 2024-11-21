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
    plt.scatter(data['Dist_true_cm'], data['Dist_calculated'])
    plt.xlabel('Real distance (cm)')
    plt.ylabel('Estimated distance (cm)')
    #graph should start at 0,0
    plt.xlim(0, 3500)
    plt.ylim(0, 6000)

    # set tick marks to be the same
    plt.xticks(range(0, 3500, 500))
    plt.yticks(range(0, 6000, 500))

    # x and y scaling should be the same
    plt.gca().set_aspect('equal', adjustable='box')

    # plot the optimal line as well in red
    plt.plot([0, 3000], [0, 3000], 'r')
    plt.show()


def calculate_distance_with_rtt(rtt: int) -> int:
    return int(rtt * SPEED_OF_LIGHT_METERS_PER_SECOND * 100 / 1e+12 / 2)

def add_distance_with_rtt(df: pd.DataFrame):
    df['Dist_calculated'] = df['RTT'].apply(func=calculate_distance_with_rtt)

def main():
    df = load_multiple_files(get_data_file_names('.out'))
    add_distance_with_rtt(df)
    print(df.head(100))
    plot_real_distance_vs_estimated_distance(df)






if __name__ == '__main__':
    main()