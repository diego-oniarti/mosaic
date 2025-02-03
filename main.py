import pathlib
import argparse
import time
from PIL import Image
from place_points import get_fracture_image
from tqdm import tqdm

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

    pathlib.Path("out").mkdir(exist_ok=True)

    # Parse arguments
    args = parser.parse_args()

    filename = args.filepath
    threshold = args.threshold
    line_size = args.line_size
    n_points = args.n_points
    timeout = args.timeout
    no_timeout = args.no_timeout

    start = time.time()
    colors, ids = get_fracture_image(filename, threshold, line_size, n_points,
                                     args.show, timeout, no_timeout)

    height = colors.shape[0]
    width = colors.shape[1]

    with tqdm(total=width*height, desc="Drawing borders", leave=False) as pbar:
        for y in range(height):
            for x in range(width):
                center_id = ids[y, x]
                different = 0
                for i in range(-1, 2):
                    if different > 2:
                        break
                    for j in range(-1, 2):
                        xoff = x+i
                        yoff = y+j
                        if not (xoff >= 0 and yoff >= 0 and xoff < width and yoff < height):
                            continue
                        if (ids[yoff, xoff] != center_id).any():
                            different += 1
                        if different > 2:
                            break

                if different > 0:
                    darken = [1, 0.8, 0.6, 0.5][different]
                    r = colors[y, x][0]
                    g = colors[y, x][1]
                    b = colors[y, x][1]
                    colors[y, x] = (r*darken, g*darken, b*darken)

                pbar.update()

    end = time.time()

    final_image = Image.fromarray(colors)
    final_image.save(args.output+".png")
    final_image.show()

    print(f"finished in {end-start}")
