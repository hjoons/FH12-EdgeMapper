import os
import cv2
from frame_helpers import convert_to_bgra_if_required
from pyk4a import PyK4APlayback
import numpy as np
from scipy import interpolate
import argparse

import h5py

# print mkv recording length
def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

argsparser = argparse.ArgumentParser()
argsparser.add_argument("--path", help="path to mkv file")
argsparser.add_argument("--out", help="name of output file")
args = argsparser.parse_args()

# Load an MKV file
if os.path.exists(args.path):
    mkv_file = args.path
else:
    print("That path doesn't exist bruh lol")
    exit()

playback = PyK4APlayback(mkv_file)
playback.open()
playback.seek(1) # in microseconds, aka 5 seconds
info(playback) # prints the recording length

# print(type(playback))
# Create a new HDF5 file
file = h5py.File(args.out, 'w')

# Create datasets for RGB images and depth maps
rgb_images = file.create_dataset('images', (0, 480, 640, 3), maxshape=(None, 480, 640, 3))
depth_images = file.create_dataset('depths', (0, 480, 640), maxshape=(None, 480, 640))

i = 0
while True:
    try:
        capture = playback.get_next_capture()

        if capture.color is not None:
            # img_color = cv2.resize(cv2.cvtColor(convert_to_bgra_if_required(0, capture.color), cv2.COLOR_BGR2RGB)[80:720, 446: 926, 0:3], (480, 640))
            img_color = cv2.cvtColor(convert_to_bgra_if_required(0, capture.color), cv2.COLOR_BGR2RGB)[120:600, 320:960, 0:3]

            # Append the RGB image to the dataset
            rgb_images.resize(i + 1, axis=0)
            rgb_images[i] = img_color

        if capture.depth is not None:
            # img_depth = cv2.resize(capture.transformed_depth[80:720, 446: 926], (480, 640))
            img_depth = capture.transformed_depth[120:600, 320:960]

            ## Interpolation
            # Append the depth map to the dataset
            invalid_mask = img_depth == 0

            # Create coordinates for valid and invalid pixels
            rows, cols = np.indices(img_depth.shape)
            valid_coords = np.column_stack((rows[~invalid_mask], cols[~invalid_mask]))
            invalid_coords = np.column_stack((rows[invalid_mask], cols[invalid_mask]))

            # Values of valid pixels
            valid_values = img_depth[~invalid_mask]

            # Use griddata to interpolate missing values (linear)
            zi = interpolate.griddata(valid_coords, valid_values, invalid_coords, method='linear', fill_value=0)

            # Replace invalid values with interpolated values
            img_depth[invalid_mask] = zi

            # Inpaint the rest if interpolation didn't get it
            mask = (img_depth == 0).astype(np.uint8)
            if np.any(mask):
                # print(np.count_nonzero(mask))
                img_depth = cv2.inpaint(img_depth, mask, 2, flags=cv2.INPAINT_TELEA)
            
            depth_images.resize(i + 1, axis=0)
            depth_images[i] = img_depth

        i += 1
        print(f'frame {i} done')

        # key = cv2.waitKey(0)
        # if key != -1:
        #     break
    except EOFError:
        break

print(i)

# Close the HDF5 file
file.close()