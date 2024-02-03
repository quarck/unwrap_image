# pip install opencv-python
# pip install imageio

import imageio.v3 as iio
import numpy as np
import cv2
import sys

default_step = 64


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


if __name__ == "__main__":
    if len(sys.argv) != 14:
        print("Invalid usage!")
        print("Usage:")
        print(f"python {sys.argv[0]}  input_file output_file " +
              "width_cells height_cells padding_cells " +
              "top_left_x..y top_right_x..y bottom_right_x..y bottom_left_x..y")
        print("Note: 'something_x..y' means two separate arguments for x & y coordinates")
        exit(0)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    width = int(sys.argv[3])
    height = int(sys.argv[4])
    pad = int(sys.argv[5])

    top_left = [int(sys.argv[6]), int(sys.argv[7])]
    top_right = [int(sys.argv[8]), int(sys.argv[9])]
    bottom_right = [int(sys.argv[10]), int(sys.argv[11])]
    bottom_left = [int(sys.argv[12]), int(sys.argv[13])]

    step = default_step
    print(f"Unwrap so that [tl: {top_left}, tr: {top_right}, br: {bottom_right}, bl:{bottom_left}] of {input_file}" +
          f" becomes an aligned {width * step}x{height * step} rectangle in {output_file} (with padding of {pad * step})")

    src = iio.imread(input_file)
    res = unwrap_image(
        src,
        width, height, pad,
        src_frame=[top_left, top_right, bottom_right, bottom_left]
    )
    iio.imwrite(output_file, res)

    print("Done")
