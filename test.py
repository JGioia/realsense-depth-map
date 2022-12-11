## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import operator

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# global vars
depth_image = None
position = None
pose_frame = None
marker_x = 0
marker_y = 0

# Note: cropping determined by testing
# depth image has wider field of view than color
# vertical corrections were very successful
# horizontal corrections were mostly successful
depth_width = int(640 * 0.63)
depth_height = int(480 * 0.63)


def convert_mm_to_feet(mm):
    return mm * 0.00328084


def crop_to_middle(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


# Note: comparison to ruler shows that vertical distance is off by factor of .86 (returned distance is too small)
# horizontal distance is off by an acceptable 1.01
def pixel_to_position(x, y, dist, vertical_angle, horizontal_angle):
    # euler transform: z then y
    half_y = depth_height / 2
    half_x = depth_width / 2

    z_angle = ((x - half_x) / half_x) * horizontal_angle
    y_angle = ((y - half_y) / half_y) * vertical_angle
    # print("z angle", z_angle)
    # print("y angle", y_angle)

    cos_z = np.cos(z_angle)
    sin_z = np.sin(z_angle)
    cos_y = np.cos(y_angle)
    sin_y = np.sin(y_angle)

    z_rotation = np.matrix([[cos_z, -sin_z, 0],
                            [sin_z, cos_z, 0],
                            [0, 0, 1]])
    y_rotation = np.matrix([[cos_y, 0, sin_y],
                            [0, 1, 0],
                            [-sin_y, 0, cos_y]])

    unit_vector = np.matrix([[1], [0], [0]])

    rotated_vector = np.matmul(y_rotation, np.matmul(z_rotation, unit_vector))

    result = rotated_vector * dist

    return result


def mouse_event(event, x, y, flags, param):
    global marker_y, marker_x
    # x = x - 640
    if event == cv2.EVENT_LBUTTONDOWN:
        marker_x = x
        marker_y = y
        # print(y, x)
        # print(pose_frame)
        # print(len(depth_image))
        # print(depth_image[y, x])
        dist = convert_mm_to_feet(depth_image[y, x])
        print("Object is", dist, "feet away")
        object_position = pixel_to_position(x, y, dist, 21 * np.pi / 180, 34.5 * np.pi / 180)
        print(object_position)


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        pose_frame = frames.get_pose_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # position = pose_frame.get_pose_data().translation

        depth_image = crop_to_middle(depth_image, (depth_height, depth_width))

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # add marker to depth colormap
        color = (0, 255, 0)
        markerType = cv2.MARKER_CROSS
        markerSize = 15
        thickness = 2
        cv2.drawMarker(depth_colormap, (marker_x, marker_y), color, markerType, markerSize, thickness)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.setMouseCallback('RealSense', mouse_event)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
