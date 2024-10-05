import pathlib
import argparse
import time
from PIL import Image, ImageDraw
import math
import numpy as np
from place_points import get_points

def get_average_color(image, bbox):
    """
    Calculate the average color of the pixels in the bounding box of the area.

    :param image: The original image
    :param bbox: The bounding box (x_min, y_min, x_max, y_max)
    :return: A tuple representing the average color (R, G, B)
    """
    cropped_region = image.crop(bbox)
    cropped_pixels = np.array(cropped_region)
    
    # Compute the average color, ignoring any potential alpha channel
    avg_color = np.mean(cropped_pixels[:, :, :3], axis=(0, 1)).astype(int)  # Average over RGB channels
    return tuple(avg_color)

def create_image_with_squares(image_path, points, output_image_path, canvas_color=(255, 255, 255)):
    """
    Create an image with rotated squares for each point, using the average color of the area under the square.

    :param image_path: Path to the original image
    :param points: List of tuples (x, y, angle) representing the points and their associated angles
    :param output_image_path: Path to save the output image
    :param canvas_color: The background color of the canvas
    """
    # Load the original image
    original_image = Image.open(image_path)

    if original_image is None:
        return

    # Compute square size based on the image area and number of points
    # square_size = 0.8 * math.sqrt(original_image.width * original_image.height / len(points))
    # square_size = int(square_size)

    # Create a blank canvas
    canvas = Image.new('RGBA', (original_image.width, original_image.height), canvas_color)

    for point in points:
        x = point.x
        y = canvas.height-point.y
        angle = point.angle
        square_size = math.sqrt(point.size)*0.85
        square_size = int(square_size)

        # Define the bounding box of the square before rotation
        half_size = square_size // 2
        bbox = (max(0, x - half_size), max(0, y - half_size), min(original_image.width, x + half_size), min(original_image.height, y + half_size))

        # Get the average color of the area under the square
        color = get_average_color(original_image, bbox)

        # Create a square with the average color
        square = Image.new('RGBA', (square_size, square_size), (255, 0, 0, 0))
        draw = ImageDraw.Draw(square)
        if square_size > canvas.width:
            square_size = 4
        draw.rectangle([(0, 0), (square_size, square_size)], fill=color)

        # Rotate the square by the given angle
        rotated_square = square.rotate(angle, expand=True)

        # Calculate the top-left corner to paste the rotated square
        top_left_x = int(x - rotated_square.width / 2)
        top_left_y = int(y - rotated_square.height / 2)

        # Paste the rotated square onto the canvas
        canvas.paste(rotated_square, (top_left_x, top_left_y), rotated_square)

    # Save the final image
    canvas.save(output_image_path)
    canvas.show()


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

    pathlib.Path("/out").mkdir(exist_ok=True)

    # Parse arguments
    args = parser.parse_args()

    filename = args.filepath
    threshold = args.threshold
    line_size = args.line_size
    n_points = args.n_points
    timeout = args.timeout
    no_timeout = args.no_timeout

    start = time.time()
    points = get_points(filename, threshold, line_size, n_points, args.show, timeout, no_timeout)
    create_image_with_squares(filename, points, args.output+".png", (0, 0, 0))
    end = time.time()

    print(f"finished in {end-start}")
