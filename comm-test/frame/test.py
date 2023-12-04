import matplotlib.pyplot as plt
import os
import cv2
import pyk4a
from pyk4a import PyK4APlayback

mkv_file = 'ecj1204_2023-11-7.mkv'

playback = PyK4APlayback(mkv_file)
playback.open()
playback.seek(5000000)


capture = playback.get_next_capture()

x = capture.transformed_depth

print("Done")
