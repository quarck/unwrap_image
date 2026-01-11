# pip install opencv-python
# pip install imageio

import imageio.v3 as iio
import numpy as np
import cv2
import sys
import math

default_step = 32
default_pad = 5

def detect_bounding_box(image_path):
     # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, which will be our grid
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    return box, rect

def unwrap_image(source, area_width, area_height, area_pad, src_frame, cell_size=None):
    if cell_size is None:
        cell_size = default_step

    src_frame = np.array(src_frame)
    source = source.copy()

    dst_tri = np.array(
        [
            [area_pad * cell_size, area_pad * cell_size],
            [area_pad * cell_size + area_width * cell_size, area_pad * cell_size + 0],
            [area_pad * cell_size + area_width * cell_size, area_pad * cell_size + area_height * cell_size],
            [area_pad * cell_size + 0, area_pad * cell_size + area_height * cell_size],
        ])

    warp_mat = cv2.getPerspectiveTransform(src_frame.astype(np.float32), dst_tri.astype(np.float32))

    dst = np.zeros(((area_width + 2 * area_pad) * cell_size, (area_height + 2 * area_pad) * cell_size, 3))

    return cv2.warpPerspective(source, warp_mat, (dst.shape[0], dst.shape[1]), dst=dst, borderValue=(255, 255, 255))

def unwrap_image_direct(source, px_area_width, px_area_height, pad_x, pad_y, src_frame):

    src_frame = np.array(src_frame)
    source = source.copy()

    dst_tri = np.array(
        [
            [pad_x,                  pad_y],
            [pad_x + px_area_width,  pad_y + 0],
            [pad_x + px_area_width,  pad_y + px_area_height],
            [pad_x + 0,              pad_y + px_area_height],
        ])

    warp_mat = cv2.getPerspectiveTransform(src_frame.astype(np.float32), dst_tri.astype(np.float32))

    dst = np.zeros((round(px_area_width + 2 * pad_x), round(px_area_height + 2 * pad_y), 3))

    return cv2.warpPerspective(source, warp_mat, (dst.shape[0], dst.shape[1]), dst=dst, borderValue=(255, 255, 255))


def print_usage_n_die():
    print("Invalid usage!")
    print("Usage:")
    print(f"python {sys.argv[0]}  input_file output_file [size width_cells height_cells] [pad n] box x0 y0 x1 y1 x2 y2 x3 y3")
    print("Note: box is a set of points in the following order: top left, top right, buttom right, buttom left")
    exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print_usage_n_die()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    args = sys.argv[3:]

    width = None
    height = None
    pad = default_pad
    step = default_step
    box = None

    while len(args):
        if args[0] == 'size':
            width = int(args[1])
            height = int(args[2])
            args = args[3:]
            continue

        if args[0] == 'pad':
            pad = int(args[1])
            args = args[2:]
            continue

        if args[0] == 'box':
            box = [ (int(args[1]), int(args[2])),
                (int(args[3]),  int(args[4])),
                (int(args[5]), int(args[6])),
                (int(args[7]), int(args[8]))
            ]
            args = args[9:]
            continue

        print_usage_n_die()

    if box is None:
        print_usage_n_die()

    original_width = math.sqrt(math.pow(box[0][0] - box[1][0], 2) + math.pow(box[0][1] - box[1][1], 2))
    original_heigh = math.sqrt(math.pow(box[1][0] - box[2][0], 2) + math.pow(box[1][1] - box[2][1], 2))
    original_ratio = original_width / original_heigh

    src = iio.imread(input_file)

    print("Bounding box is ", box)

    if width is not None and height is not None:

        print(f"Unwrap so that {box}" +
            f" becomes an aligned {width * step}x{height * step} rectangle in {output_file} (with padding of {pad * step})")
        new_ratio = width / height
        print(f"Original ratio: {original_ratio}:1, new_ratio: {new_ratio}:1  ratio over ratio:{original_ratio/new_ratio}")

        res = unwrap_image(
            src, width, height, pad, src_frame=box
        )
        iio.imwrite(output_file, res)

    else: 
        # size in cells is not given - but we don't even need it (at all!)

        pad_x = pad_y = pad * default_step
        if original_ratio >= 1.0:
            pad_x = round(pad_x * original_ratio)
        else:
            pad_y = round(pad_y / original_ratio)

        print(f"Unwrap so that {box}" +
            f" becomes an aligned {round(original_width)}x{round(original_heigh)} rectangle in {output_file}" + 
            f" (with padding {pad_x}px/{pad_y}px along x/y")

        res = unwrap_image_direct(
            src, original_width, original_heigh, pad_x, pad_y, src_frame=box
        )
        iio.imwrite(output_file, res)

    print("Done")
