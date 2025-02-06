import math
import subprocess
import os
import pathlib
import argparse
import time
from place_points import get_fracture_image
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_colors(filename):
    colors = dict()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            id = int(parts[0])
            r, g, b = map(int, parts[1].split(','))
            colors[id] = (b, g, r)

    return colors


def interpolate_color(a, b, d):
    return tuple(np.round(np.array(a) * (1 - d) + np.array(b) * d).astype(int))


def process_frame(frame_file, a_colors, b_colors, n_points, d):
    interpolated_colors = dict()
    for id in range(n_points):
        a_color = a_colors[id]
        b_color = b_colors[id]
        interpolated_colors[id] = interpolate_color(a_color, b_color, d)

    image_path = os.path.join('frames', frame_file)
    image = cv2.imread(image_path)
    out_image_path = os.path.join('frames_color', frame_file)

    for id in range(n_points):
        id_r = id & 0xff
        id_g = (id >> 8) & 0xff
        id_b = (id >> 16) & 0xff
        id_col = np.array([id_b, id_g, id_r])
        interpolated_color = interpolated_colors[id]

        mask = cv2.inRange(image, id_col, id_col)
        image[mask > 0] = interpolated_color

    cv2.imwrite(out_image_path, image)


def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t**3
    else:
        return 1 - 4 * (1 - t)**3


def reverse_ease_in_out_cubic(t):
    return 1-ease_in_out_cubic(t)


def custom_ease(t):
    if t < 0.5:
        return pow(t*2, 2 / 3) / 2
    else:
        return 1 - (math.sqrt((1-t)*2)/2)  # Steep increase at the end


def custom_ease2(t):
    return 1 - math.sqrt(1-t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Turn an image into a mosaic")

    parser.add_argument('filepath', type=str, help="Path to the input image")
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help="Threshold for the edge detection.")
    parser.add_argument('-l', '--line_size', type=int, default=0, help="How much to bolden the edge lines")
    parser.add_argument('-n', '--n_points', type=int, default=500, help="Number of pieces in the mosaic")
    parser.add_argument('-s', '--show', action='store_true', help="Shows the window with the voronoi status")
    parser.add_argument('-T', '--timeout', type=int, default=60, help="Upperbound to the processing time. Expressed in seconds")
    parser.add_argument('-N', '--no_timeout', action='store_true', help="Ignore the timeout. Can make the program run foorever if the voronoi doesn't converge on the image")
    parser.add_argument('-o', '--output', type=str, default="out/final", help="Path to the output image. Do not specify extension")
    parser.add_argument('-i', '--interpolate', action='store_true', help="Initializas the centroids with last image's centroids")

    pathlib.Path("out").mkdir(exist_ok=True)

    # Parse arguments
    args = parser.parse_args()

    filename = args.filepath
    threshold = args.threshold
    line_size = args.line_size
    n_points = args.n_points
    timeout = args.timeout
    no_timeout = args.no_timeout
    interpolate = args.interpolate

    start = time.time()
    colors = get_fracture_image(filename, threshold, line_size,
                                n_points, args.show, timeout,
                                no_timeout, interpolate)

    Image.fromarray(colors).show()
    height = colors.shape[0]
    width = colors.shape[1]

    with open("movements.txt", "r") as file:
        line = file.readline().strip()
        movements = list(map(float, line.split(' ')))

    if not interpolate:
        exit(0)

    a_colors = load_colors("colors_a.txt")
    b_colors = load_colors("colors_b.txt")

    tot_movements = 0
    for movement in movements:
        tot_movements += movement

    video_duration = 5.0  # secondi

    # rendi la somma dei movements=1
    for i in range(len(movements)):
        movements[i] = movements[i]/tot_movements

    frame_files = sorted([f for f in os.listdir("frames") if f.endswith('.png')])
    num_images = len(frame_files)

    cumulative_weights = np.cumsum(movements)
    normalizaed_time = cumulative_weights / cumulative_weights[-1]
    eased_time = np.array([custom_ease(t) for t in normalizaed_time])
    eased_weights = np.diff(eased_time, prepend=0)
    eased_weights /= np.sum(eased_weights)

    movements = eased_weights

    with open("input.txt", "w") as file:
        for i, frame_name in enumerate(frame_files):
            file.write(f"file 'frames_color/{frame_name}'\n")
            file.write(f"duration {(movements[i]*video_duration):.3f}\n")

    with ThreadPoolExecutor() as executor:
        futures = []
        d = 0
        for i, frame_file in enumerate(frame_files):
            # d = i / (num_images - 1)
            futures.append(executor.submit(process_frame, frame_file, a_colors, b_colors, n_points, d))
            d += movements[i]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            future.result()

    end = time.time()
    print(f"finished in {end-start}")

    try:
        subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-i', 'input.txt',
            '-vsync', 'vfr',
            'frames_color/output.mp4'
        ], check=True)
        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output.mp4',
            '-vf', "fps=30",
            '-vsync', 'cfr',
            'frames_color/output_cfr.mp4'
        ], check=True)
        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output_cfr.mp4',
            '-vf', "reverse",
            'frames_color/reversed.mp4'
        ], check=True)
        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output_cfr.mp4',
            '-i', 'frames_color/reversed.mp4',
            '-filter_complex', "[0:v][1:v]concat=n=2:v=1:[v]",
            '-map', "[v]",
            'frames_color/combined.mp4'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}\n")
