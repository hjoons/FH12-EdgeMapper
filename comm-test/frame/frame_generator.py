# import cv as mkv_reader
import matplotlib.pyplot as plt
from matplotlib import image
import os
import cv2
from frame_helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")
          
# Load an MKV file
if os.path.exists("C:/Users/tiani/SeniorDesign/ecj1204_2023-11-7.mkv"): # !!!!! Change this path to the correct directory/mkv file
    mkv_file = "C:/Users/tiani/SeniorDesign/ecj1204_2023-11-7.mkv"
else:
    print("That path doesn't exist bruh lol")
    exit()

playback = PyK4APlayback(mkv_file)
playback.open()
playback.seek(5000000)
info(playback)

while True:

    try:
        capture = playback.get_next_capture()
        if capture.color is None:
            print("no capture color")
        if capture.color is not None:
            # print("capture.color", capture.color)

            # not the correct crop and NYU2 dataset size
            #cv2.imshow("Color", convert_to_bgra_if_required(0, capture.color))
            
            # 640 x 480
            img_color = cv2.resize(cv2.cvtColor(convert_to_bgra_if_required(0, capture.color), cv2.COLOR_BGR2RGB)[240:720, 286: 926, 0:3], (640, 480))
            cv2.imshow("Color", img_color)
            

        if capture.depth is None:
            print("no capture depth")
        if capture.depth is not None:
            # print("capture depth", capture.depth[450,450])

            # not the correct crop and NYU2 dataset size
            # cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))

            # 640 x 480
            img_depth = cv2.resize(capture.transformed_depth[240:720, 286: 926], (640, 480))
            cv2.imshow("Depth", img_depth)
            # cv2.imshow(capture.depth)
        key = cv2.waitKey(0)
        if key == -1:
            break
    except EOFError:
        print("sorry no color or depth in this capture")
        break  

# Release OpenCV windows
cv2.destroyAllWindows()
playback.close()
