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


def lerp(a, b, x):
    return a*(1-x) + b*x


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
    image_path = os.path.join('frames', frame_file)
    image = cv2.imread(image_path)
    out_image_path = os.path.join('frames_color', frame_file)
    if os.path.isfile(out_image_path):
        return

    interpolated_colors = dict()
    for id in range(n_points):
        a_color = a_colors[id]
        b_color = b_colors[id]
        interpolated_colors[id] = interpolate_color(a_color, b_color, d)

    for id in range(n_points):
        id_r = id & 0xff
        id_g = (id >> 8) & 0xff
        id_b = (id >> 16) & 0xff
        id_col = np.array([id_b, id_g, id_r])
        interpolated_color = interpolated_colors[id]

        # Create a mask for the current color
        mask = cv2.inRange(image, id_col, id_col)

        # Replace the color in the image
        image[mask > 0] = interpolated_color

        # Create an inner border using erosion
        kernel = np.ones((3, 3), np.uint8)  # Kernel for erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        inner_border_mask = mask - eroded_mask

        # Darken the inner border pixels
        darken_factor = 0.75  # Adjust this value to control the darkness
        image[inner_border_mask > 0] = image[inner_border_mask > 0] * darken_factor

    cv2.imwrite(out_image_path, image)


def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t**3
    else:
        return 1 - 4 * (1 - t)**3


def reverse_ease_in_out_cubic(t):
    return 1-ease_in_out_cubic(t)


def custom_ease(t):
    # if t < 0.5:
    #     return pow(t*2, 1 / 2) / 2
    # else:
    #     return 1 - (pow((1-t)*2, 1/2)/2)  # Steep increase at the end
    n = 5
    if t < 0.5:
        return pow(2*t, 1/n)/2
    else:
        return 1 - pow((1-t)*2, 1/n)/2


def custom_ease2(t):
    return 1 - pow(1-t, 2/5)


def custom_ease3(t):
    n = 3
    if t < 0.1:
        return pow(t*10, 1/n)/10
    else:
        return 1-pow((1-t)/0.9, 1/n)*0.9


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
    parser.add_argument('-L', '--limit_dist', action='store_true', help="Limits the average distance moved by the points")

    pathlib.Path("out").mkdir(exist_ok=True)
    pathlib.Path("frames").mkdir(exist_ok=True)
    pathlib.Path("frames_color").mkdir(exist_ok=True)

    for file in os.listdir("frames"):
        os.remove(os.path.join("frames", file))
    for file in os.listdir("frames_color"):
        os.remove(os.path.join("frames_color", file))

    # Parse arguments
    args = parser.parse_args()

    filename = args.filepath
    threshold = args.threshold
    line_size = args.line_size
    n_points = args.n_points
    timeout = args.timeout
    no_timeout = args.no_timeout
    interpolate = args.interpolate
    limit_dist = args.limit_dist

    start = time.time()
    colors = get_fracture_image(filename, threshold, line_size,
                                n_points, args.show, timeout,
                                no_timeout, interpolate, limit_dist)

    # Image.fromarray(colors).show()

    image = Image.open(filename)
    width, height = image.width, image.height

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

    video_duration = len(movements)/60  # secondi

    # rendi la somma dei movements=1
    for i in range(len(movements)):
        movements[i] = movements[i]/tot_movements

    frame_files = sorted([f for f in os.listdir("frames") if f.endswith('.png')])
    num_images = len(frame_files)

    cumulative_weights = np.cumsum(movements)
    normalizaed_time = cumulative_weights / cumulative_weights[-1]
    eased_time = np.array([custom_ease3(t) for t in normalizaed_time])
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
            d_ease = ease_in_out_cubic(d)
            futures.append(executor.submit(process_frame, frame_file, a_colors, b_colors, n_points, d_ease))
            d += movements[i]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            future.result()

    end = time.time()
    print(f"finished in {end-start}")

    if width % 2 != 0:
        width -= 1  # or width += 1, depending on your preference
    if height % 2 != 0:
        height -= 1  # or height += 1, depending on your preference
    try:
        subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-i', 'input.txt',
            '-vsync', 'vfr',
            '-strict', '-2',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-s', f"{width}x{height}",
            'frames_color/output.mp4'
        ], check=True)
        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output.mp4',
            '-vf', 'tpad=stop_mode=clone:stop_duration=0.1',
            'frames_color/output_padded.mp4'
        ], check=True)
        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output_padded.mp4',
            '-vf', "fps=30",
            '-vsync', 'cfr',
            '-c:v', 'libx264',
            '-strict', '-2',
            '-pix_fmt', 'yuv420p',
            '-s', f"{width}x{height}",
            'frames_color/output_cfr.mp4'
        ], check=True)

        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output_cfr.mp4',
            '-vf', "reverse",
            '-c:v', 'libx264',
            '-strict', '-2',
            '-pix_fmt', 'yuv420p',
            '-s', f"{width}x{height}",
            'frames_color/reversed.mp4'
        ], check=True)

        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/output_cfr.mp4',
            '-i', 'frames_color/reversed.mp4',
            '-filter_complex', "[0:v][1:v]concat=n=2:v=1:[v]",
            '-map', "[v]",
            '-c:v', 'libx264',
            '-strict', '-2',
            '-pix_fmt', 'yuv420p',
            '-s', f"{width}x{height}",
            'frames_color/combined.mp4'
        ], check=True)

        # Re-encode for final compatibility
        subprocess.run([
            'ffmpeg',
            '-i', 'frames_color/combined.mp4',
            '-strict', '-2',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            'frames_color/final_output.mp4',
            '-s', f"{width}x{height}"
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}\n")
