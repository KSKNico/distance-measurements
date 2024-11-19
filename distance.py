import os
import argparse
import subprocess
import time

DATA_PATH = "data"

def save_measurement(name: str, measurement: str):
    with open(os.path.join(DATA_PATH, name), 'a+') as f:
        f.write(measurement)

    

def launch_measurement(name: str):
    idf_command = "ftm -I".encode('utf-8')

    result = subprocess.run(["idf.py", "monitor"], subprocess.PIPE, input=idf_command)

    # sleep for 5 seconds to make sure the measurement is done
    time.sleep(5)

    return result.stdout.decode('utf-8')    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    m = launch_measurement(args.name)
    save_measurement(m)

if __name__ == '__main__':
    main()