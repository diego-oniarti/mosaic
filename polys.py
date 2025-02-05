import cv2
import numpy as np
from tqdm import tqdm


def extract_all_polygons(tile_count):
    # Load image and convert to RGB color space
    image = cv2.imread("ids.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    polygons = {}

    # NEW: Default polygon parameters
    default_size = 5  # Adjust as needed
    img_height, img_width = image.shape[:2]
    default_poly = [
        (img_width - default_size, img_height - default_size),
        (img_width, img_height - default_size),
        (img_width, img_height),
        (img_width - default_size, img_height)
    ]

    for tile_id in tqdm(range(tile_count), leave=False, desc="Extracting polygons"):
        # Get color for current tile ID (NEW: added modulo 256 for safety)
        target_color = (
            (tile_id) & 0xFF,          # R (8 bits)
            (tile_id >> 8) & 0xFF,     # G (next 8 bits)
            (tile_id >> 16) & 0xFF     # B (next 8 bits)
        )

        # Create binary mask for this color
        mask = np.all(image_rgb == target_color, axis=2).astype(np.uint8) * 255

        # NEW: Always create an entry, even for missing tiles
        if np.sum(mask) > 0:  # Check if color exists in image
            # Find external contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Simplify polygon with dynamic epsilon
                contour = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                polygon = [tuple(point[0]) for point in approx]
                polygons[tile_id] = polygon
                continue

        # Fallback for missing tiles (NEW: added to ensure completeness)
        polygons[tile_id] = default_poly

    return polygons


def write_polygons(polygons):
    with open("polygons.txt", "w") as file:
        for polygon in tqdm(polygons.items(), leave=False, desc="Writing polygons"):
            file.write(f"{polygon[0]} {' '.join(map(lambda x: str(x[0])+','+str(x[1]),polygon[1]))}\n")
