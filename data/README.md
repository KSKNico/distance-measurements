# Hardware
Two ESP-32 S3 and a ublox m8 GNSS module.  

# Software
The data was gathered by using the ESP-IDF and writing an application that uses WiFi FTM. The application can be found under the main subfolder of the parent directory. `get_idf` needs to be executed so that the correct environment is set to make the ESP-IDF work. One also needs to install the correct python packages to make the main.py data evaluation script work. Refer to the `requirements.txt`.

# Measurments

A few measurements in different conditions were conducted. The measurements are split into three categories:
1. different angles and same distance around the bicycle with one sensor attached to the back of the saddle with WiFi FTM (different_angles_saddle/)
2. different distances away from the bicycle with clear LoS with WiFi FTM (different_distances/).
3. different distances with GPS (gps/), using ublox generation 8.

Each experiment was conducted multiple times. For the gps measurements the subfolders corresponds to the subfolders in different_distances.

## Different angles saddle
The distance was 5 meters to the bicycle. Each measurement moves in 30 degree steps around the bike for a total of 12. The sensor was attached to the back of the saddle.

## Different distances
Measurements 01/ and 02/ were conducted with no obstacles in a radius of 5 meters. Measurment 03/ was conducted close (~1m) to a tall building. In all cases, both sensors were 1 meter above the ground.

## GPS
The GPS measurements were conducted alongside different_distances/. So they correspond to the respective experiments done there. The configuration of the ublox GNSS receiver was set to a refresh rate of 1 Hz and put into stationary mode.

## Evaluation
The data can be plotted with the `main.py` file. It can process all three experiments by supplying the corresponding argument (r/d/dg) and the subfolder of the experiment. The evaluation of GPS data is always done alongside the different_distances (d) experiments as the GPS was intended to be a reference technology.
Example being `python main.py dg 03` - this evaluates the FTM data in the 03 subfolder by comparing it to the GPS data in the 03 subfolder of the GPS experiment. The graphs will end up in different_distances/03/graphs/*.pdf.