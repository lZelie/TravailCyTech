#include <array>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <GL/freeglut_std.h>

#define CHECK_CUDA_ERROR(err) check_cuda_error_d(err, __FILE__, __LINE__)

enum class render_mode: std::uint8_t
{
	cpu,
	gpu,
};

constexpr std::size_t screen_x = 1024;
constexpr std::size_t screen_y = 768;
constexpr std::size_t fps_update = 500;
constexpr auto title = "RayTracer";

std::array<float4, screen_x * screen_y> pixels{};
std::size_t frame = 0;
std::size_t time_base = 0;
float scale = 3e-3f;
__device__ float ambient_light = 0.2f;

auto render_mode = render_mode::cpu;

inline void check_cuda_error_d(const cudaError err, const std::string& file, const int line)
{
	if (err != cudaError::cudaSuccess)
	{
		std::cerr << file << "(" << line << "): CUDA Runtime API error " << err << ": " << cudaGetErrorString(err) <<
			". \n";
		std::cin.get();

		std::quick_exit(err);
	}
}


void init_gl(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(screen_x, screen_y);
	glutCreateWindow(title);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, screen_x, screen_y, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}


void calculate_pixels_gpu()
{
}


void calculate_pixels_cpu()
{
	const int range = 5;
}

void calculate()
{
	frame++;

	const int time_current = glutGet(GLUT_ELAPSED_TIME);

	if (time_current - time_base > fps_update)
	{
		std::stringstream ss;
		ss << title << ": " << std::fixed << std::setprecision(3) << static_cast<float>(frame) * 1e3f / static_cast<
			float>(time_current - time_base) << " FPS";
		glutSetWindowTitle(ss.str().c_str());
		time_base = time_current;
		frame = 0;
	}

	if (render_mode == render_mode::cpu)
	{
		calculate_pixels_cpu();
	}
	else
	{
		calculate_pixels_gpu();
	}
}

void render()
{
	calculate();

	glDrawPixels(screen_x, screen_y, GL_RGBA, GL_FLOAT, pixels.data());

	glutSwapBuffers();
}

void idle()
{
	glutPostRedisplay();
}

float3 random_float3(const float min, const float max)
{
	static std::random_device rng;
	std::uniform_real_distribution<float> dist(min, max);
	return {dist(rng), dist(rng), dist(rng)};
}

float random_float(const float min, const float max)
{
	static std::random_device rng;
	std::uniform_real_distribution<float> dist(min, max);
	return dist(rng);
}


void process_normal_keys(const unsigned char key, int, int)
{
	switch (key) // NOLINT(hicpp-multiway-paths-covered)
	{
	case 27:
		std::quick_exit(0);
	case '1':
		render_mode = render_mode::cpu;
		glutPostRedisplay();
		break;
	case '2':
		render_mode = render_mode::gpu;
		glutPostRedisplay();
		break;
	default:
		break;
	}
}


int main(const int argc, char* argv[])
{
	init_gl(argc, argv);

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutKeyboardFunc(process_normal_keys);


	glutMainLoop();

	return EXIT_SUCCESS;
}
