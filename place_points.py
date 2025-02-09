import sys
import numpy as np
import math
import glfw
import OpenGL.GL as gl
import OpenGL.GLU as glu
import random
from edges import get_edges
from PIL import Image
import cv2
import time
from tqdm import tqdm


# Initialize GLFW and create a window
def init_glfw_window(width, height, title, visible):
    if not glfw.init():
        return None

    # Set window visibility based on the `visible` flag
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        return None

    glfw.make_context_current(window)
    gl.glClearColor(0.0, 0.0, 1.0, 1.0)  # Set background color to white

    # Set orthographic projection for top-down view
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height, -10, 10)  # Orthographic projection
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth test for 3D rendering

    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

    return window


class Point:
    def __init__(self, x: float, y: float, color: int):
        self.x = x
        self.y = y
        self.color = color
        self.angle = 0
        self.size = 0


def generate_random_points(num_points: int, width: int,
                           height: int) -> list[Point]:
    '''Generate random Points from -plane_size a +plane_size'''
    points = []
    for i in range(num_points):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        points.append(Point(x, y, i))
    return points


def generate_seeds(num_points: int, width: int, height: int,
                   probability_bias) -> list[Point]:
    with tqdm(total=num_points, desc="Placing seeds", leave=False) as pbar:
        points = []
        while True:
            for y in range(height):
                for x in range(width):
                    local_bias = probability_bias[y, x]
                    if local_bias == 0:
                        continue
                    if random.uniform(1, 100) < pow(local_bias,1/2)*1.1:
                        points.append(Point(x, y, len(points)))
                        pbar.update()
                    if len(points) == num_points:
                        return points


def read_seeds():
    points = []
    with open("centroids.txt", "r") as file:
        lines = file.readlines()
        for line in tqdm(lines, leave=False, desc="Reading centroids"):
            parts = line.split(' ')
            points.append(Point(float(parts[1]), float(parts[2]),
                                int(parts[0])))

    return points


def draw_cone_at_point(x: float, y: float, color: int, angle=0.0,
                       base_radius=200, height=7.0, num_slices=20,
                       slope=0.0):
    '''Disegna un cono nella posizione e rotazione indicata'''
    angle -= 45
    gl.glPushMatrix()
    gl.glTranslatef(x, y, -slope*100)  # Translate cone to (x, y) on the plane

    gl.glColor3ub(color & 0b11111111, (color >> 8) & 0b11111111, color >> 16)

    # Draw the cone using gluCylinder
    cone_quadric = glu.gluNewQuadric()
    gl.glRotatef(angle, 0, 0, 1)
    glu.gluCylinder(cone_quadric, base_radius, 0.0,
                    height+slope * 100, num_slices, 1)

    gl.glPopMatrix()


def draw_point(x, y, size=4):
    gl.glPointSize(size)  # Set the size of the point
    gl.glBegin(gl.GL_POINTS)
    gl.glVertex3f(x, y, 8)  # Specify the position of the point in 2D space
    gl.glEnd()


def interpol(a, b, x):
    return a*(1-x) + b*x

# Main rendering loop
# Ritorna la matrice del voronoi colorato e quello di ID
def get_fracture_image(path, thr, thick, n_points, visible=False, timeout=30, no_timeout=False, interpolate=False, limit_dist=False):
    edges = get_edges(path, thr, thick)
    if edges is None:
        print("Couldn't get the flowfield")
        return

    edges_image = Image.fromarray(edges)
    edges_image.save("edges.png", format="png")

    edges = np.flip(edges, 0)

    binary_edges = 255 - (edges[:, :, 3] > 0).astype(np.uint8) * 255
    distance_transform = cv2.distanceTransform(binary_edges, cv2.DIST_L2,
                                               cv2.DIST_MASK_PRECISE)
    # distance_transform = np.clip(distance_transform, 0, 50)
    distance_transform = distance_transform / np.max(distance_transform)

    dist_image = Image.fromarray(np.flip((distance_transform*255)
                                         .astype(np.uint8), 0))
    dist_image.save("dist.png", format="png")

    image = Image.open(path)
    width, height = image.width, image.height

    window = init_glfw_window(width, height, "Mosaic", visible)

    if not window:
        print("Failed to create GLFW window")
        return

    thr = 0.1
    exp = 10
    size_bias = (thr - np.clip(distance_transform, 0, thr))/thr
    size_bias = pow(size_bias, exp)

    mag_image_pixels = np.flip(size_bias*255, 0)
    mag_image_pixels = np.array([[(x, x, x) if x != 0 else (255, 0, 0) for x in row] for row in mag_image_pixels]).astype(np.uint8)
    Image.fromarray(mag_image_pixels).save("mag.png", format="png")

    if interpolate:
        points = read_seeds()
    else:
        # points = generate_seeds(n_points, width, height, size_bias)
        points = generate_random_points(n_points, width, height)

    start_time = time.time()

    # Main loop to render the scene
    finished = False
    gap_closer = 1
    min_dist = 99999
    original_min_dist = -1

    frame_count = 0
    tot_pixels = width*height

    bar = tqdm(leave=False)
    movements = []
    while not (glfw.window_should_close(window)
               or (finished and gap_closer <= 0)):
        if not no_timeout and not finished:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Time limit exceeded, stopping the loop.")
                finished = True

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for point in points:
            draw_cone_at_point(point.x, point.y, point.color,
                               base_radius=2 * (width + height))

        # Calculate centroids
        aree = dict()
        for point in points:
            aree[point.color] = [0, 0, 0]

        pixel_data = np.zeros((height, width, 3), dtype=np.uint8)
        gl.glReadPixels(0, 0, width, height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixel_data)

        frame_file_name = f"frame_{frame_count:04d}.png"
        if interpolate:
            Image.fromarray(np.flip(pixel_data, 0)).save(f"frames/{frame_file_name}", format="png")
            frame_count += 1

        bar.set_description("Update centroids")
        bar.reset(total=tot_pixels)
        min_dist_cutoff = 0.6
        for pix_y in range(0, height, 2):
            for pix_x in range(0, width, 2):
                edges_mask = edges[pix_y, pix_x]
                if edges_mask[3] != 0 and not finished:
                    continue
                D = size_bias[pix_y, pix_x] + thr/10000
                col = pixel_data[pix_y, pix_x]
                colid = (int(col[0]) & 0b11111111) + (int(col[1]) << 8)

                aree[colid][0] += pix_x * D
                aree[colid][1] += pix_y * D
                aree[colid][2] += D

                bar.update()

        is_still = True
        max_dist = 0
        total_movement = 0

        delta_pos = dict()
        for point in points:
            area = aree[point.color]
            if area[2] == 0:
                continue
            # il +0.5 fa funzionare tutto. Non sono sicuro del motivo
            new_x = area[0] / area[2] + 0.5
            new_y = area[1] / area[2] + 0.5

            dx = (new_x - point.x)
            dy = (new_y - point.y)

            dist = math.sqrt(math.pow(dx, 2)
                             + math.pow(dy, 2))

            delta_pos[point.color] = (dx, dy)

            total_movement += dist

            if dist > max_dist:
                max_dist = dist

        average_movement = total_movement / n_points
        movement_multiplier = 1
        average_movement_top = 1.5 if limit_dist else (width+height)
        if average_movement > average_movement_top:
            movement_multiplier = average_movement_top / average_movement
            average_movement = average_movement_top
        for i in delta_pos.keys():
            points[i].x += delta_pos[i][0] * movement_multiplier
            points[i].y += delta_pos[i][1] * movement_multiplier

        movements.append(average_movement)

        if min_dist > min_dist_cutoff or is_still and dist > min_dist:
            is_still = False

        if max_dist < min_dist:
            min_dist = max_dist
            if original_min_dist == -1:
                original_min_dist = min_dist

        bar.write(f"{(pow((original_min_dist - min_dist)/(original_min_dist-min_dist_cutoff), 50)*100):.2f}%", end="\r")

        if finished:
            gap_closer -= 1

        if not finished and is_still:
            finished = True

        glfw.swap_buffers(window)
        glfw.poll_events()

    bar.close()
    print("Finished.")
    glfw.swap_buffers(window)

    # calcola la media dei colori per regione
    aree_colori = dict()
    for point in points:
        aree_colori[point.color] = [0, 0, 0, 0]  # R G B count

    pixel_data = np.zeros((height, width, 3), dtype=np.uint8)
    gl.glReadPixels(0, 0, width, height,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixel_data)
    pixel_data = np.flip(pixel_data, 0)

    for pix_y in range(height):
        for pix_x in range(width):
            col = pixel_data[pix_y, pix_x]
            colid = (int(col[0]) & 0b11111111) + (int(col[1]) << 8)

            image_color = image.getpixel((pix_x, pix_y))
            aree_colori[colid][0] += image_color[0]
            aree_colori[colid][1] += image_color[1]
            aree_colori[colid][2] += image_color[2]
            aree_colori[colid][3] += 1

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    colors_file_name = "colors_a.txt" if not interpolate else "colors_b.txt"
    with open(colors_file_name, "w") as file:
        for point in tqdm(points, leave=False, desc="Drawing image"):
            area_colore = aree_colori[point.color]
            if area_colore[3] != 0:
                r = int(area_colore[0]/area_colore[3])
                g = int(area_colore[1]/area_colore[3])
                b = int(area_colore[2]/area_colore[3])
            else:
                r = 0
                g = 0
                b = 0

            draw_cone_at_point(point.x, point.y, ((b << 16) + (g << 8) + r),
                               base_radius=2 * (width + height))

            file.write(f"{point.color} {r},{g},{b}\n")

    final_image_pixels = np.zeros((height, width, 3), dtype=np.uint8)
    gl.glReadPixels(0, 0, width, height,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, final_image_pixels)
    final_image_pixels = np.flip(final_image_pixels, 0)

    glfw.terminate()

    if not interpolate:
        with open("centroids.txt", "w") as centroids_file:
            for point in tqdm(points, leave=False, desc="File dump"):
                centroids_file.write(f"{point.color} {point.x} {point.y}\n")

    with open("movements.txt", "w") as file:
        file.write(" ".join(map(str, movements)))

    return final_image_pixels


if __name__ == "__main__":
    filename = sys.argv[1]
    threshold = float(sys.argv[2])
    line_size = int(sys.argv[3])
    n_points = int(sys.argv[4])
    points = get_fracture_image(filename, threshold, line_size, n_points, True)
    print(points)
