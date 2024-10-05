import sys
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw

# Vertex shader source
vertex_shader_code = """
#version 330
in vec2 position;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = (position + 1.0) / 2.0;
}
"""

# Fragment shader for Sobel filter
fragment_shader_sobel_code = """
#version 330
uniform sampler2D tex;
in vec2 TexCoord;
out vec4 FragColor;

const float kernelX[9] = float[](-1, 0, 1, -2, 0, 2, -1, 0, 1);
const float kernelY[9] = float[](-1, -2, -1, 0, 0, 0, 1, 2, 1);
uniform float threshold;

void main() {
    vec2 texSize = textureSize(tex, 0);
    float texOffset = 1.0 / texSize.x;

    vec3 gradientX = vec3(0.0);
    vec3 gradientY = vec3(0.0);

    // Apply Sobel filter
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec3 sample = texture(tex, TexCoord + vec2(i, j) * texOffset).rgb;
            int index = (i + 1) * 3 + (j + 1);
            gradientX += kernelX[index] * sample;
            gradientY += kernelY[index] * sample;
        }
    }

    // Compute gradient magnitude
    float magnitude = length(vec2(gradientX.r, gradientY.r));

    // Thresholding: if the magnitude is greater than threshold, mark as edge
    if (magnitude > threshold) {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // White for edges
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);  // Transparent for non-edges
    }
}
"""

# Fragment shader for dilation (thickening edges)
fragment_shader_dilation_code = """
#version 330
uniform sampler2D tex;
in vec2 TexCoord;
out vec4 FragColor;
uniform int edge_thickness;

void main() {
    vec2 texSize = textureSize(tex, 0);
    vec2 texelSize = 1.0 / texSize;

    vec4 result = vec4(0.0);

    // Dilation: Check surrounding pixels within the 'edge_thickness' radius
    for (int x = -edge_thickness; x <= edge_thickness; x++) {
        for (int y = -edge_thickness; y <= edge_thickness; y++) {
            if (length(vec2(x, y)) <= float(edge_thickness)) {
                vec4 neighbor = texture(tex, TexCoord + vec2(x, y) * texelSize);
                result = max(result, neighbor);  // If any neighboring pixel is an edge, keep it
            }
        }
    }

    FragColor = result;
}
"""

# Function to create a shader program
def create_shader_program(vertex_shader_code, fragment_shader_code):
    vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    return compileProgram(vertex_shader, fragment_shader)

# Function to load an image and convert it to a format suitable for OpenGL
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(image, dtype=np.uint8)
    return img_data, image.width, image.height

# Function to create an OpenGL texture from image data
def setup_texture(image_data, width, height):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture

# Function to set up a quad for rendering
def setup_quad():
    quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0
    ], dtype=np.float32)

    quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * quad_vertices.itemsize, ctypes.c_void_p(0))

    return VAO

# Function to set up a framebuffer for offscreen rendering
def setup_fbo(width, height):
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Framebuffer not complete")
        return None

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, texture

# Main rendering function
def get_edges(image_path: str, threshold: float, edge_thickness: int):
    '''Reads an image and returns a npArray with white edges and transparent non-edges'''
    # Initialize GLFW
    if not glfw.init():
        return

    # Load image and get dimensions
    image_data, width, height = load_image(image_path)

    # Create a window with the initial size based on image dimensions
    window_width = width
    window_height = height
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(window_width, window_height, "Edge Detection", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Load texture
    texture = setup_texture(image_data, width, height)

    # Create shader programs
    sobel_shader_program = create_shader_program(vertex_shader_code, fragment_shader_sobel_code)
    dilation_shader_program = create_shader_program(vertex_shader_code, fragment_shader_dilation_code)

    # Setup quad for rendering
    VAO = setup_quad()

    # Setup framebuffer for offscreen rendering
    fbo, fbo_texture = setup_fbo(width, height)

    # Step 1: Render Sobel filter to FBO (with thresholding)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(sobel_shader_program)
    glUniform1f(glGetUniformLocation(sobel_shader_program, "threshold"), threshold)  # Pass threshold
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # Step 2: Render from FBO (apply edge thickness via dilation)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(dilation_shader_program)
    glUniform1i(glGetUniformLocation(dilation_shader_program, "edge_thickness"), edge_thickness)  # Pass thickness
    glBindTexture(GL_TEXTURE_2D, fbo_texture)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # Get the final image
    final_image = np.zeros((height, width, 4), dtype=np.uint8)
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, final_image)

    # Clean up resources
    glDeleteFramebuffers(1, [fbo])
    glDeleteTextures(1, [fbo_texture])
    glfw.terminate()

    return final_image

if __name__ == "__main__":
    image_path = sys.argv[1]  # Replace with your image path
    threshold = float(sys.argv[2])  # Replace with your threshold value
    edge_thickness = int(sys.argv[3])  # Replace with your edge thickness value
    final_image = get_edges(image_path, threshold, edge_thickness)
    if final_image is not None:
        result_image = Image.fromarray(final_image)
        result_image.show()
