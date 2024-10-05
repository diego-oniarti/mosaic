import sys
import cv2
import numpy as np
from PIL import Image

def get_edges(image_path: str, threshold: float, edge_thickness: int):
    '''Reads an image, applies Sobel edge detection, and returns a binary image with white edges and transparent background.'''

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Step 1: Apply Sobel filter to detect edges
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel in X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel in Y direction

    # Compute gradient magnitude (hypotenuse of x and y)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient magnitude to [0, 255] for better visualization
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Step 2: Apply threshold to get binary edge image
    _, binary_edges = cv2.threshold(sobel_magnitude, (1.0-threshold)*255, 255, cv2.THRESH_BINARY)

    # Step 3: Apply dilation to thicken edges based on edge_thickness
    if edge_thickness > 0:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        dilated_edges = cv2.dilate(binary_edges, kernel)
    else:
        dilated_edges = binary_edges

    # Step 4: Convert the binary edges into a 4-channel RGBA image (white edges, transparent background)
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # Set white color for edges
    rgba_image[:, :, 0] = dilated_edges  # Red channel
    rgba_image[:, :, 1] = dilated_edges  # Green channel
    rgba_image[:, :, 2] = dilated_edges  # Blue channel

    # Set alpha channel to 255 where edges are detected, else 0 for transparency
    rgba_image[:, :, 3] = dilated_edges

    return rgba_image


if __name__ == "__main__":
    image_path = sys.argv[1]  # Replace with your image path
    threshold = float(sys.argv[2])  # Replace with your threshold value (0-255 for cv2)
    edge_thickness = int(sys.argv[3])  # Replace with your edge thickness value

    # Get the final image with edges and transparent background
    final_image = get_edges_cv2(image_path, threshold, edge_thickness)

    if final_image is not None:
        # Convert to PIL Image to show or save
        result_image = Image.fromarray(final_image)
        result_image.show()  # Display image or use .save() to save it to disk
