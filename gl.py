import pygame
import numpy
import glm
import pyassimp
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


pygame.init()
clock = pygame.time.Clock()

screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)


vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

uniform mat4 theMatrix;
uniform vec3 light;

out float intensity;

void main() 
{
    intensity = dot(normal, normalize(light));
    gl_Position = theMatrix * vec4(position.x, position.y, position.z, 1.0f);
}
"""

fragment_shader = """
#version 460
layout (location = 0) out vec4 fragColor;
in float intensity;

void main()
{
  fragColor = vec4(1.0f, 0.0f, 1.0f, 1.0f) * intensity;
}
"""

shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

scene = pyassimp.load('./goku.obj')
print(scene)


def glize(node):
    for mesh in node.meshes:
        vertex_data = numpy.hstack([
            numpy.array(mesh.vertices, dtype=numpy.float32),
            numpy.array(mesh.normals, dtype=numpy.float32),
            numpy.array(mesh.texturecoords[0], dtype=numpy.float32),
        ])

        index_data = numpy.hstack(
            numpy.array(mesh.faces, dtype=numpy.int32),
        )

        vertex_buffer_object = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes,
                     vertex_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              9 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              9 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE,
                              9 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes,
                     index_data, GL_STATIC_DRAW)

        glUniform3f(
            glGetUniformLocation(shader, "light"),
            -10, 10, 5
        )

        glDrawElements(GL_TRIANGLES, len(index_data), GL_UNSIGNED_INT, None)

    for child in node.children:
        glize(child)

# vertex_data = numpy.array([
#     -0.5, -0.5,  0.5,    1, 0, 0,
#     0.5, -0.5,  0.5,    1, 1, 0,
#     0.5,  0.5,  0.5,    0, 0, 1,
#     -0.5,  0.5,  0.5,    1, 1, 0,
#     -0.5, -0.5, -0.5,    1, 0, 0,
#     0.5, -0.5, -0.5,    0, 1, 0,
#     0.5,  0.5, -0.5,    0, 0, 1,
#     -0.5,  0.5, -0.5,    1, 1, 0
# ], dtype=numpy.float32)

# index_data = numpy.array([
#     # front
#     0, 1, 2,
#     2, 3, 0,
#     # right
#     1, 5, 6,
#     6, 2, 1,
#     # back
#     7, 6, 5,
#     5, 4, 7,
#     # left
#     4, 0, 3,
#     3, 7, 4,
#     # bottom
#     4, 5, 1,
#     1, 0, 4,
#     # top
#     3, 2, 6,
#     6, 7, 3
# ], dtype=numpy.uint32)

# vertex_array_object = glGenVertexArrays(1)
# glBindVertexArray(vertex_array_object)

# vertex_buffer_object = glGenBuffers(1)
# glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
# glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

# element_buffer_object = glGenBuffers(1)
# glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
# glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes,
#              index_data, GL_STATIC_DRAW)


# glVertexAttribPointer(
#     0,
#     3,
#     GL_FLOAT,
#     GL_FALSE,
#     6 * 4,
#     ctypes.c_void_p(0)
# )
# glEnableVertexAttribArray(0)

# glVertexAttribPointer(
#     1,
#     3,
#     GL_FLOAT,
#     GL_FALSE,
#     6 * 4,
#     ctypes.c_void_p(3 * 4)
# )
# glEnableVertexAttribArray(1)

i = glm.mat4()


def createTheMatrix(movement):

    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate_x = glm.rotate(i, glm.radians(movement[0]), glm.vec3(1, 0, 0))
    rotate_y = glm.rotate(i, glm.radians(movement[1]), glm.vec3(0, 1, 0))
    scale = glm.scale(i, glm.vec3(1, 1, 1))

    model = translate * rotate_x * rotate_y * scale
    view = glm.lookAt(glm.vec3(0, 0, 1),
                      glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    projection = glm.perspective(glm.radians(45), 800/600, 0.1, 1000)

    return projection * view * model


glViewport(0, 0, 800, 600)

glEnable(GL_DEPTH_TEST)


running = True
movement = [0, 0, 0]
while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearcolor = (0.5, 1.0, 0.5, 1.0)

    glUseProgram(shader)
    theMatrix = createTheMatrix(movement)

    theMatrixLocation = glGetUniformLocation(shader, 'theMatrix')

    glUniformMatrix4fv(
        theMatrixLocation,
        1,
        GL_FALSE,
        glm.value_ptr(theMatrix)
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            if event.key == pygame.K_2:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            if event.key == pygame.K_d:
                movement[1] += 10
            if event.key == pygame.K_a:
                movement[1] -= 10
            if event.key == pygame.K_s:
                movement[0] -= 10
            if event.key == pygame.K_w:
                movement[0] += 10

    # glDrawArrays(GL_TRIANGLES, 0, 3)
    # glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
    glize(scene.rootnode)

    pygame.display.flip()

    clock.tick(100)
