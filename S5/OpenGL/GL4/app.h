//
// Created by cytech on 24/09/24.
//

#ifndef GL2_APP_H
#define GL2_APP_H

#include <memory>
#include <GL/glew.h>
#include <vector>
#include "shader_class.h"
#include "vao.h"
#include "glm/vec4.hpp"
#include <memory>

#include "fbo.h"
#include "texture.h"
#include "glm/vec3.hpp"

namespace gl3 {

    class app {
        const char *TITLE = "Hello OpenGL!";
        static int width;
        static int height;
        static glm::vec4 rotation;

        static std::unique_ptr<gl3::vao> vaoSuzanne;
        std::unique_ptr<gl3::vbo> vboSuzanne;
        std::unique_ptr<gl3::vbo> vboSuzanneSmooth;
        std::unique_ptr<gl3::vbo> vboQuad;
        static std::unique_ptr<gl3::shader_class> program_texture_colored;
        static bool smoothed;
        static bool textured;
        static std::unique_ptr<texture> tex;
        static glm::vec3 light_pos;
        static glm::vec3 cam_pos;
        static bool light;
        static bool ambient;
        static bool diffuse;
        static bool specular;
        static bool is_gouraud;
        static int mouse_button;


        static std::vector<GLfloat> suzanne_mesh_data;

        static void render_suzanne(const std::unique_ptr<gl3::vao>& suzanne);
        static void displayFunc();

        static void keyboard(unsigned char key, int x, int y);

        static void updateMVP(glm::vec3 translation);

        static void reshapeFunc(int newWidth, int newHeight);

        static enum class post
        {
            clear,
            blur,
            grayscale,
            pixelize,
            sobel,
        } post_process;

    public:
        static std::unique_ptr<shader_class> program_texture_no_colored;
        static std::unique_ptr<shader_class> program_no_texture_colored_gouraud;
        static std::unique_ptr<shader_class> program_no_texture_colored_phong;
        static bool coloredText;
        static glm::vec4 Ka;
        static glm::vec4 Kd;
        static glm::vec4 Ks;
        static std::unique_ptr<shader_class> program_cell_shading_outline;
        static std::unique_ptr<shader_class> program_cell_shading_color;
        static bool cell_shading;
        static std::unique_ptr<vao> vaoQuad;
        static std::unique_ptr<shader_class> program_shader_tex;
        static std::unique_ptr<fbo> framebuffer;
        static std::unique_ptr<shader_class> program_post_process_no;
        static std::unique_ptr<fbo> framebuffer2;
        static std::unique_ptr<shader_class> program_post_process_blur;
        static std::unique_ptr<shader_class> program_post_process_grayscale;
        static std::unique_ptr<shader_class> program_post_process_pixelize;
        static std::unique_ptr<shader_class> program_post_process_sobel;
        void init(int argc, char **argv);

        void start();

        ~app();

        static void motionFunc(int x, int y);
        static void mouseFunc(int button, int state, int x, int y);

        static std::unique_ptr<gl3::vao> vaoSuzanneSmooth;
    };

} // gl2

#endif //GL2_APP_H
