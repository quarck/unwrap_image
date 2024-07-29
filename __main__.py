# pip install opencv-python
# pip install imageio

import imageio.v3 as iio
import numpy as np
import cv2
import sys

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
    box = np.int0(box)

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


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Invalid usage!")
        print("Usage:")
        print(f"python {sys.argv[0]}  input_file output_file width_cells height_cells ")
        exit(0)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    width = int(sys.argv[3])
    height = int(sys.argv[4])    

    pad = default_pad
    step = default_step

    if len(sys.argv) >= 14 and sys.argv[5] == 'box':
        box = [ (int(sys.argv[6]), int(sys.argv[7])), 
                (int(sys.argv[8]), int(sys.argv[9])), 
                (int(sys.argv[10]), int(sys.argv[11])), 
                (int(sys.argv[12]), int(sys.argv[13]))
            ]
    else:
        box, _ = detect_bounding_box(input_file)

    print("Bounding box is ", box)
    print(f"Unwrap so that {box}" +
          f" becomes an aligned {width * step}x{height * step} rectangle in {output_file} (with padding of {pad * step})")

    src = iio.imread(input_file)
    res = unwrap_image(
        src, width, height, pad, src_frame=box
    )
    iio.imwrite(output_file, res)

    print("Done")
