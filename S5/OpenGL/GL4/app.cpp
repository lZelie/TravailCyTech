//
// Created by cytech on 24/09/24.
//

#include "app.h"

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <memory>
#include <memory>
#include <sstream>

#include "fbo.h"
#include "shader_class.h"
#include "texture.h"
#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

namespace gl3
{
    std::vector<GLfloat> load_obj(const char* filename, bool smooth = false)
    {
        std::vector<glm::vec4> vertices;
        std::vector<glm::vec2> text_coords;
        std::vector<glm::vec3> normals;
        std::vector<std::array<GLushort, 3>> faces;
        {
            std::ifstream in{filename, std::ios::in};
            if (!in)
            {
                std::cerr << "Failed to open file " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            std::string line;
            while (std::getline(in, line))
            {
                if (line.substr(0, 2) == "v ")
                {
                    std::istringstream s{line.substr(1)};
                    glm::vec4 v4;
                    s >> v4.x;
                    s >> v4.y;
                    v4.y = -v4.y;
                    s >> v4.z;
                    v4.w = 1.0f;
                    vertices.push_back(v4);
                }
                else if (line.substr(0, 2) == "vt")
                {
                    std::istringstream s{line.substr(2)};
                    glm::vec2 v4;
                    s >> v4.x;
                    s >> v4.y;
                    text_coords.push_back(v4);
                }
                else if (line.substr(0, 2) == "vn")
                {
                    std::istringstream s{line.substr(2)};
                    glm::vec3 v4;
                    s >> v4.x;
                    s >> v4.y;
                    v4.y = -v4.y;
                    s >> v4.z;
                    normals.push_back(v4);
                }
                else if (line.substr(0, 2) == "f ")
                {
                    std::istringstream s{line.substr(1)};
                    std::array<std::array<GLushort, 3>, 3> v4{};
                    char splash;
                    s >> v4[0][0];
                    s >> splash;
                    s >> v4[0][1];
                    s >> splash;
                    s >> v4[0][2];
                    s >> v4[1][0];
                    s >> splash;
                    s >> v4[1][1];
                    s >> splash;
                    s >> v4[1][2];
                    s >> v4[2][0];
                    s >> splash;
                    s >> v4[2][1];
                    s >> splash;
                    s >> v4[2][2];
                    faces.insert(std::end(faces), std::begin(v4), std::end(v4));
                }
            }
        }
        std::map<GLushort, std::vector<std::array<GLushort, 3>>> grouped;
        for (const auto& face : faces)
        {
            grouped[face[0]].push_back(face);
        }

        std::map<GLushort, glm::vec3> vertex_normals;
        for (const auto& [vert, faces] : grouped)
        {
            glm::vec3 sum{};
            for (const auto& face : faces)
            {
                sum += normals[face[2] - 1];
            }
            vertex_normals[vert] = glm::vec3(sum.x / faces.size(), sum.y / faces.size(), sum.z / faces.size());
        }
        std::vector<GLfloat> mesh_data;
        for (const auto& face : faces)
        {
            auto v0 = vertices[face[0] - 1];
            auto vt0 = text_coords[face[1] - 1];
            auto n0 = smooth ? vertex_normals[face[0]] : normals[face[2] - 1];
            mesh_data.insert(mesh_data.end(), {
                                 v0.x, v0.y, v0.z, v0.w, vt0.x, vt0.y, n0.x, n0.y, n0.z,
                             });
        }
        return mesh_data;
    }


    std::unique_ptr<vao> app::vaoSuzanne;
    std::unique_ptr<vao> app::vaoSuzanneSmooth;
    std::unique_ptr<vao> app::vaoQuad;
    std::unique_ptr<fbo> app::framebuffer;
    std::unique_ptr<fbo> app::framebuffer2;
    std::vector<GLfloat> app::suzanne_mesh_data;
    std::unique_ptr<shader_class> app::program_texture_colored;
    std::unique_ptr<shader_class> app::program_texture_no_colored;
    std::unique_ptr<shader_class> app::program_no_texture_colored_gouraud;
    std::unique_ptr<shader_class> app::program_no_texture_colored_phong;
    std::unique_ptr<shader_class> app::program_cell_shading_outline;
    std::unique_ptr<shader_class> app::program_cell_shading_color;
    std::unique_ptr<shader_class> app::program_shader_tex;
    std::unique_ptr<shader_class> app::program_post_process_no;
    std::unique_ptr<shader_class> app::program_post_process_blur;
    std::unique_ptr<shader_class> app::program_post_process_grayscale;
    std::unique_ptr<shader_class> app::program_post_process_pixelize;
    std::unique_ptr<shader_class> app::program_post_process_sobel;
    std::unique_ptr<texture> app::tex;
    glm::vec4 app::Ka{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 app::Kd{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 app::Ks{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec3 app::light_pos{0.0f, 0.0f, 30.0f};
    bool app::smoothed = false;
    bool app::textured = false;
    bool app::coloredText = true;
    int app::width = 1024;
    int app::height = 768;
    glm::vec4 app::rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 app::cam_pos{0.0f, 0.0f, 20.0f};
    bool app::light = false;
    bool app::ambient = false;
    bool app::diffuse = false;
    bool app::specular = false;
    bool app::is_gouraud = false;
    bool app::cell_shading = false;
    app::post app::post_process = post::clear;
    int app::mouse_button = GLUT_LEFT_BUTTON;

    void app::init(int argc, char** argv)
    {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
        glutInitWindowSize(width, height);
        glutCreateWindow(TITLE);
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        glClearColor(1.0, 1.0, 1.0, .0);
        glViewport(0, 0, width, height);
        if (const GLenum err = glewInit(); err != GLEW_OK)
        {
            std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Using GLEW: " << glewGetString(GLEW_VERSION) << std::endl;

        glEnable(GL_DEPTH_TEST);

        suzanne_mesh_data = load_obj("../objects/suzanne.obj");
        vaoSuzanne = std::make_unique<vao>();
        vaoSuzanne->bind();

        vboSuzanne = std::make_unique<vbo>(suzanne_mesh_data);

        program_texture_colored = std::make_unique<shader_class>("../shaders/vertex/model.vert",
                                                                 "../shaders/fragments/model.frag");
        program_texture_no_colored = std::make_unique<shader_class>("../shaders/vertex/model.vert",
                                                                    "../shaders/fragments/model2.frag");
        program_no_texture_colored_gouraud = std::make_unique<shader_class>("../shaders/vertex/model_gouraud.vert",
                                                                            "../shaders/fragments/model.frag");
        program_no_texture_colored_phong = std::make_unique<shader_class>("../shaders/vertex/model_phong.vert",
                                                                          "../shaders/fragments/model_phong.frag");

        program_post_process_no = std::make_unique<shader_class>("../shaders/vertex/no_process.vert",
                                                                 "../shaders/fragments/no_process.frag");

        program_post_process_blur = std::make_unique<shader_class>("../shaders/vertex/no_process.vert",
                                                                 "../shaders/fragments/blur_prost.frag");

        program_cell_shading_outline = std::make_unique<shader_class>("../shaders/vertex/cell_outline.vert",
                                                                      "../shaders/fragments/cell_outline.frag");
        program_cell_shading_color = std::make_unique<shader_class>("../shaders/vertex/cell_color.vert",
                                                                    "../shaders/fragments/cell_color.frag");

        program_post_process_grayscale = std::make_unique<shader_class>("../shaders/vertex/no_process.vert",
                                                                    "../shaders/fragments/grayscale_post.frag");

        program_post_process_pixelize = std::make_unique<shader_class>("../shaders/vertex/no_process.vert",
                                                                    "../shaders/fragments/pixelize_post.frag");

        program_post_process_sobel = std::make_unique<shader_class>("../shaders/vertex/no_process.vert",
                                                                    "../shaders/fragments/sobel_post.frag");

        vaoSuzanne->linkAttrib(*vboSuzanne, 0, 4, GL_FLOAT, 9 * sizeof(GLfloat), nullptr);
        vaoSuzanne->linkAttrib(*vboSuzanne, 1, 2, GL_FLOAT, 9 * sizeof(GLfloat),
                               reinterpret_cast<GLvoid*>(4 * sizeof(GLfloat)));
        vaoSuzanne->linkAttrib(*vboSuzanne, 2, 3, GL_FLOAT, 9 * sizeof(GLfloat),
                               reinterpret_cast<GLvoid*>(6 * sizeof(GLfloat)));

        vaoSuzanne->unbind();

        suzanne_mesh_data = load_obj("../objects/suzanne.obj", true);
        vaoSuzanneSmooth = std::make_unique<vao>();
        vaoSuzanneSmooth->bind();
        vboSuzanneSmooth = std::make_unique<vbo>(suzanne_mesh_data);
        vaoSuzanneSmooth->linkAttrib(*vboSuzanneSmooth, 0, 4, GL_FLOAT, 9 * sizeof(GLfloat), nullptr);
        vaoSuzanneSmooth->linkAttrib(*vboSuzanneSmooth, 1, 2, GL_FLOAT, 9 * sizeof(GLfloat),
                                     reinterpret_cast<GLvoid*>(4 * sizeof(GLfloat)));
        vaoSuzanneSmooth->linkAttrib(*vboSuzanneSmooth, 2, 3, GL_FLOAT, 9 * sizeof(GLfloat),
                                     reinterpret_cast<GLvoid*>(6 * sizeof(GLfloat)));

        vaoSuzanneSmooth->unbind();

        vaoQuad = std::make_unique<vao>();
        vaoQuad->bind();
        const std::vector quad_vertices = {-1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f};
        vboQuad = std::make_unique<vbo>(quad_vertices);
        vaoQuad->linkAttrib(*vboQuad, 0, 2, GL_FLOAT, 0, nullptr);
        vaoQuad->unbind();
        program_shader_tex = std::make_unique<shader_class>("../shaders/vertex/shader_tex.vert",
                                                            "../shaders/fragments/shader_tex.frag");

        framebuffer = std::make_unique<fbo>(width, height);
        framebuffer2 = std::make_unique<fbo>(512, 512);

        glutDisplayFunc(displayFunc);
        glutReshapeFunc(reshapeFunc);
        glutMotionFunc(motionFunc);
        glutMouseFunc(mouseFunc);
        glutIdleFunc([] { glutPostRedisplay(); });

        glutKeyboardFunc(keyboard);
    }


    void app::motionFunc(int x, int y)
    {
        if (mouse_button == GLUT_LEFT_BUTTON)
        {
            float angleY = abs((y - height / 2.0f) / (height / 2.0f));
            float angleX = abs((x - width / 2.0f) / (width / 2.0f));
            rotation.w = -(angleX + angleY);
            rotation.x = ((y - height / 2.0f) / (height / 2.0f));
            rotation.y = ((x - width / 2.0f) / (width / 2.0f));
        }
        else if (mouse_button == GLUT_RIGHT_BUTTON)
        {
            light_pos.x = -(x - width / 2.0f);
            light_pos.y = (y - height / 2.0f);
        }
        glutPostRedisplay();
    }

    void app::mouseFunc(int button, int state, int x, int y)
    {
        if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) mouse_button = GLUT_LEFT_BUTTON;
        if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) mouse_button = GLUT_RIGHT_BUTTON;
    }


    void app::reshapeFunc(int newWidth, int newHeight)
    {
        width = newWidth;
        height = newHeight;
        glutPostRedisplay();
    }

    float what_time_is_it()
    {
        static float start = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
        const float now = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
        return now - start;
    }

    void app::render_suzanne(const std::unique_ptr<gl3::vao>& suzanne)
    {
        if (!cell_shading)
        {
            if (coloredText)
            {
                if (light)
                {
                    if (is_gouraud)
                    {
                        program_no_texture_colored_gouraud->activate();
                    }
                    else
                    {
                        program_no_texture_colored_phong->activate();
                    }
                }
                else
                {
                    program_texture_colored->activate();
                }
            }
            else
            {
                program_texture_no_colored->activate();
            }
        }
        else
        {
            program_cell_shading_outline->activate();
        }
        updateMVP({0.0f, 0.0f, 0.0f});
        glDrawArrays(GL_TRIANGLES, 0, suzanne_mesh_data.size() / 9);
        suzanne->unbind();
        if (cell_shading)
        {
            glClear(GL_DEPTH_BUFFER_BIT);
            program_cell_shading_color->activate();
            suzanne->bind();
            updateMVP({0.0f, 0.0f, 0.0f});
            glDrawArrays(GL_TRIANGLES, 0, suzanne_mesh_data.size() / 9);
            suzanne->unbind();
        }
    }

    void app::displayFunc()
    {
        glViewport(0, 0, framebuffer2->width, framebuffer2->height);
        framebuffer2->bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        program_shader_tex->activate();
        glUniform1f(1, what_time_is_it());
        vaoQuad->bind();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glViewport(0, 0, framebuffer->width, framebuffer->height);
        framebuffer->bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        framebuffer2->bind_texture();
        const auto& suzanne = smoothed ? vaoSuzanneSmooth : vaoSuzanne;
        suzanne->bind();

        render_suzanne(suzanne);

        glViewport(0, 0, width, height);
        fbo::bind_screen();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        framebuffer->bind_texture();
        switch (post_process)
        {
        case post::blur:
            program_post_process_blur->activate();
            glUniform1f(1, 5.0f);
            break;
        case post::pixelize:
            program_post_process_pixelize->activate();
            glUniform1f(1, 0.01f);
            break;
        case post::sobel:
            program_post_process_sobel->activate();
            break;
        case post::grayscale:
            program_post_process_grayscale->activate();
            break;
        case post::clear:
        default:
            program_post_process_no->activate();
            break;
        }
        vaoQuad->bind();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glutSwapBuffers();
    }

    void app::keyboard(unsigned char key, int, int)
    {
        switch (key)
        {
        case '\033':
        case 'q':
        case 'Q':
            glutExit();
            break;
        case 'S':
        case 's':
            smoothed = !smoothed;
            glutPostRedisplay();
            break;
        case 't':
        case 'T':
            textured = !textured;
            tex = textured ? std::make_unique<texture>(256, 256) : std::make_unique<texture>();
            tex->tex_unit(*program_texture_colored, "tex", 0);
            glutPostRedisplay();
            break;
        case 'c':
        case 'C':
            coloredText = !coloredText;
            glutPostRedisplay();
            break;
        case 'v':
        case 'V':
            cell_shading = !cell_shading;
            glutPostRedisplay();
            break;
        case 'l':
        case 'L':
            light = !light;
            glutPostRedisplay();
            break;
        case 'a':
        case 'A':
            ambient = !ambient;
            if (ambient)
            {
                Ka = glm::vec4(0.2f);
            }
            else
            {
                Ka = glm::vec4(0.0f);
            }
            glutPostRedisplay();
            break;
        case 'd':
        case 'D':
            diffuse = !diffuse;
            if (diffuse)
            {
                Kd = glm::vec4(1.0f);
            }
            else
            {
                Kd = glm::vec4(0.0f);
            }
            glutPostRedisplay();
            break;
        case 'p':
        case 'P':
            specular = !specular;
            if (specular)
            {
                Ks = glm::vec4(1.0f);
            }
            else
            {
                Ks = glm::vec4(0.0f);
            }
            glutPostRedisplay();
            break;
        case 'g':
        case 'G':
            is_gouraud = !is_gouraud;
            glutPostRedisplay();
            break;
        case 'y':
        case 'Y':
            post_process = post_process == post::blur ? post::clear : post::blur;
            glutPostRedisplay();
            break;
        case 'u':
        case 'U':
            post_process = post_process == post::grayscale ? post::clear : post::grayscale;
            glutPostRedisplay();
            break;
        case 'i':
        case 'I':
            post_process = post_process == post::pixelize ? post::clear : post::pixelize;
            glutPostRedisplay();
            break;
        case 'o':
        case 'O':
            post_process = post_process == post::sobel ? post::clear : post::sobel;
            glutPostRedisplay();
            break;
        default:
            break;
        }
    }

    app::~app()
    {
        vaoSuzanne.reset();
        program_texture_colored.reset();
    }

    void app::start()
    {
        glutMainLoop();
    }

    void app::updateMVP(glm::vec3 translation)
    {
        glm::mat4 projection = glm::perspective(50.0f, static_cast<float>(width) / height, 1.0f, 100.0f);
        glm::mat4 view = lookAt(cam_pos, glm::vec3{0.0f, 0.0f, 0.0f},
                                glm::vec3{0.0f, 1.0f, 0.0f});
        glm::mat4 model{1.0};
        model = translate(model, translation);
        model = rotate(model, rotation.w, glm::vec3(rotation.x, rotation.y, rotation.z));
        model = scale(model, glm::vec3(1.5f));
        glUniformMatrix4fv(3, 1, GL_FALSE, &model[0][0]);
        glUniformMatrix4fv(4, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(5, 1, GL_FALSE, &projection[0][0]);
        glUniform4fv(6, 1, &Ka[0]);
        glUniform4fv(7, 1, &Kd[0]);
        glUniform4fv(8, 1, &Ks[0]);
        glUniform1f(9, 1.0f);
        glUniform3fv(10, 1, &light_pos[0]);
        glUniform3fv(11, 1, &cam_pos[0]);
    }
} // gl2
