# import cv as mkv_reader
import matplotlib.pyplot as plt
from matplotlib import image
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
from frame_helpers import colorize, convert_to_bgra_if_required
import pyk4a
from pyk4a import PyK4APlayback, SeekOrigin
import numpy as np

# Function to display depth map
def display_depth_map(depth_frame):
    # Convert the depth frame to a numpy array
    depth_data = depth_frame.data
    #depth_data = depth_data.reshape((depth_frame.height, depth_frame.width))
    
    # Normalize depth values for visualization
    # depth_data = (depth_data / 4096.0 * 255).astype(np.uint8)
    
    # Display the depth map
    cv2.imshow("Depth Map", depth_data)
    cv2.waitKey(0)

def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")
          
# Load an MKV file
if os.path.exists("C:/Users/tiani/SeniorDesign/ecj1204_2023-11-7.mkv"):
    mkv_file = "C:/Users/tiani/SeniorDesign/ecj1204_2023-11-7.mkv"
else:
    print("That path doesn't exist bruh lol")
    exit()

playback = PyK4APlayback(mkv_file)
#start_timestamp = 5000000
#playback.seek(5000000, origin=SeekOrigin.BEGIN)


while True:
#     # Read a capture frame
#     capture = playback.get_next_capture()
#     if not capture:
#         break

    # Get the color and depth frames

    #print(capture)
    # color_frame = capture.color
    # depth_frame = capture.depth
    # print(color_frame)
    # print(color_frame.data)
    # print(depth_frame)

    # Display the color
    playback = PyK4APlayback(mkv_file)
    playback.open()
    playback.seek(5000000)
    info(playback)
    print(playback.length)
    try:
        capture = playback.get_next_capture()
        print(type(capture.color.data))
        print(type(capture.depth))
        if capture is None:
            print("no capture")
        #if capture.color is not None:

        cv2.imshow("Color", convert_to_bgra_if_required(pyk4a.ImageFormat.COLOR_MJPG, capture.color))
        # plt.imshow(capture.color)
        #if capture.depth is not None:
        cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
        # plt.imshow(capture.depth)
        key = cv2.waitKey(10)
        if key != -1:
            break
    except EOFError:
        print("sorry no color or depth in this capture")
        break  

        

    # Display the color frame
        cv2.imshow("Color Frame", color_image)
        cv2.waitKey(1)  # Add a small delay to allow OpenCV to update the display


    # print(color_frame)
    # print(color_frame.data)
    # print(depth_frame)
    #cv2.imshow("Color Frame", color_frame.data)

    # Check if it's an Azure Kinect recording (depth frame is available)
    
    #if depth_frame is not None:
        #display_depth_map(depth_frame)

    # Exit on key press (e.g., 'q')
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

# Release OpenCV windows
cv2.destroyAllWindows()
playback.close()
