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

    return window


class Point:
    def __init__(self, x: float, y: float, color: int):
        self.x = x
        self.y = y
        self.color = color
        self.angle = 0
        self.size = 0


def generate_random_points(num_points: int, width: int, height: int) -> list[Point]:
    '''Generate random Points from -plane_size a +plane_size'''
    points = []
    for i in range(num_points):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        points.append(Point(x, y, i))
    return points


def draw_cone_at_point(x: int, y: int, color: int, angle: int, base_radius=200, height=7.0, num_slices=4):
    '''Disegna un cono nella posizione e rotazione indicata'''
    angle -= 45
    gl.glPushMatrix()
    gl.glTranslatef(x, y, 0)  # Translate cone to (x, y) on the plane

    gl.glColor3ub(color & 0b11111111, color >> 8, 0)

    # Draw the cone using gluCylinder
    cone_quadric = glu.gluNewQuadric()
    gl.glRotatef(angle, 0, 0, 1)
    glu.gluCylinder(cone_quadric, base_radius, 0.0, height, num_slices, 1)

    gl.glPopMatrix()


def draw_point(x, y, size=4):
    gl.glPointSize(size)  # Set the size of the point
    gl.glBegin(gl.GL_POINTS)
    gl.glVertex3f(x, y, 8)  # Specify the position of the point in 2D space
    gl.glEnd()


# Main rendering loop
def get_points(path, thr, thick, n_points, visible=False, timeout=30, no_timeout=False):
    edges = get_edges(path, thr, thick)
    if edges is None:
        print("Couldn't get the flowfield")
        return

    # I contorni vengono salvati solo per scopo di debug. Non c'è alcun motivo pratico
    edges_image = Image.fromarray(edges)
    edges_image.save("edges.png", format="png")

    edges = np.flip(edges, 0)

    distance_transform = cv2.distanceTransform(edges[:, :, 3], cv2.DIST_L2, 0)

    # Compute gradients of the distance transform
    grad_x = cv2.Sobel(distance_transform, cv2.CV_64F, 1, 0, ksize=31)
    grad_y = cv2.Sobel(distance_transform, cv2.CV_64F, 0, 1, ksize=31)

    # Normalize the gradient vectors to unit vectors (flow direction)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    flow_x = grad_x / (magnitude + 1e-8)  # Add a small epsilon to avoid division by zero
    flow_y = grad_y / (magnitude + 1e-8)

    image = Image.open(path)
    width, height = image.width, image.height

    window = init_glfw_window(width, height, "Mosaic", visible)

    if not window:
        print("Failed to create GLFW window")
        return

    points = generate_random_points(n_points, width, height)

    start_time = time.time()

    # Main loop to render the scene
    finished = False
    gap_closer = 1
    while not (glfw.window_should_close(window) or (finished and gap_closer <= 0)):
        if not no_timeout and not finished:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Time limit exceeded, stopping the loop.")
                finished = True

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for point in points:
            grad_x = flow_x[int(point.y), int(point.x)]
            grad_y = flow_y[int(point.y), int(point.x)]

            angle = math.asin(grad_y) / math.pi * 180
            if grad_x < 0:
                angle = 180 - angle

            point.angle = int(angle)
            draw_cone_at_point(point.x, point.y, point.color, int(angle), 100 * (width + height))

        # Calculate centroids
        aree = dict()
        for point in points:
            aree[point.color] = [0, 0, 0]

        pixel_data = np.zeros((height, width, 3), dtype=np.uint8)
        gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixel_data)

        for pix_y in range(height):
            for pix_x in range(width):
                edges_mask = edges[pix_y, pix_x]
                if edges_mask[3] != 0 and not finished:
                    continue
                col = pixel_data[pix_y, pix_x]
                colid = (int(col[0]) & 0b11111111) + (int(col[1]) << 8)

                aree[colid][0] += pix_x
                aree[colid][1] += pix_y
                aree[colid][2] += 1

        is_still = True
        for point in points:
            area = aree[point.color]
            if area[2] == 0:
                continue
            # il +0.5 fa funzionare tutto. Non sono sicuro del motivo
            new_x = area[0] / area[2] + 0.5
            new_y = area[1] / area[2] + 0.5
            if is_still and math.sqrt(math.pow(new_x - point.x, 2) + math.pow(new_y - point.y, 2)) > 0.5:
                is_still = False
            point.x = new_x
            point.y = new_y
            point.size = area[2]

        if finished:
            gap_closer -= 1

        if not finished and is_still:
            finished = True
            print("Finished.")

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

    return points


if __name__ == "__main__":
    filename = sys.argv[1]
    threshold = float(sys.argv[2])
    line_size = int(sys.argv[3])
    n_points = int(sys.argv[4])
    points = get_points(filename, threshold, line_size, n_points, True)
    print(points)
