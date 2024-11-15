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
constexpr auto title = "Bugs";

std::array<float4, screen_x * screen_y> pixels{};
std::size_t frame = 0;
std::size_t time_base = 0;
float scale = 3e-3f;
constexpr int range = 1;
constexpr int survive_low = 3;
constexpr int survive_high = 4;
constexpr int birth_low = 3;
constexpr int birth_high = 3;
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

__global__ void calculate_pixels_gpu_1(const float4* pixels_in_d, float4* pixels_out_d)
{
	const std::size_t index_x = threadIdx.x + blockIdx.x * blockDim.x;
	const std::size_t index_y = threadIdx.y + blockIdx.y * blockDim.y;
	const std::size_t index_pixel = index_y * screen_x + index_x;

	const int min_x = static_cast<int>(index_x) - range;
	const int min_y = static_cast<int>(index_y) - range;

	int livings = 0;
	if (pixels_in_d[index_pixel].y == 1.0f)
	{
		livings -= 1;
	}

	for (int square_x = min_x; square_x < min_x + (2 * range + 1); ++square_x)
	{
		for (int square_y = min_y; square_y < min_y + (2 * range + 1); ++square_y)
		{
			const int corrected_x = square_x >= 0
				                        ? (square_x < static_cast<int>(screen_x)
					                           ? square_x
					                           : square_x -
					                           static_cast<int>(screen_x))
				                        : static_cast<int>(screen_x) + square_x;
			const int corrected_y = square_y >= 0
				                        ? (square_y < static_cast<int>(screen_y)
					                           ? square_y
					                           : square_y -
					                           static_cast<int>(screen_y))
				                        : static_cast<int>(screen_y) + square_y;

			const std::size_t index_pixel_square = corrected_y * screen_x + corrected_x;

			if (pixels_in_d[index_pixel_square].y == 1.0f)
			{
				livings += 1;
			}
		}
	}

	if (pixels_in_d[index_pixel].y == 0.0f && livings >= birth_low && livings <= birth_high)
	{
		pixels_out_d[index_pixel] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
	}
	else if (pixels_in_d[index_pixel].y == 1.0f && (livings < survive_low || livings > survive_high))
	{
		pixels_out_d[index_pixel] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
}


void calculate_pixels_gpu()
{
	/*static int grid = 1;

	if (grid == 1)
	{*/
	float4 *grid1, *grid2;

	cudaMalloc(&grid1, screen_x * screen_y * sizeof(float4));
	cudaMalloc(&grid2, screen_x * screen_y * sizeof(float4));
	cudaMemcpy(grid1, pixels.data(), screen_x * screen_y * sizeof(float4), cudaMemcpyHostToDevice);
	// Using grid1 to grid2
	calculate_pixels_gpu_1 <<< dim3(screen_x / 16, screen_y / 16), dim3(16, 16) >>>(grid1, grid2);
	cudaMemcpy(pixels.data(), grid2, screen_x * screen_y * sizeof(float4), cudaMemcpyDeviceToHost);

	/*grid = 2;*/
	/*}
	else
	{
		// Using grid2 to grid1
		calculate_pixels_gpu_1 <<< dim3(screen_x / 16, screen_y / 16), dim3(16, 16) >>>(grid2, grid1);
		cudaMemcpy(pixels.data(), grid1, screen_x * screen_y * sizeof(float4), cudaMemcpyDeviceToHost);

		grid = 1;
	}*/
}


void calculate_pixels_cpu()
{
	for (std::size_t index_x = 0; index_x < screen_x; ++index_x)
	{
		for (std::size_t index_y = 0; index_y < screen_y; ++index_y)
		{
			const std::size_t index_pixel = index_y * screen_x + index_x;

			const int min_x = static_cast<int>(index_x) - range;
			const int min_y = static_cast<int>(index_y) - range;

			int livings = 0;
			if (pixels[index_pixel].y == 1.0f)
			{
				livings -= 1;
			}

			for (int square_x = min_x; square_x < min_x + (2 * range + 1); ++square_x)
			{
				for (int square_y = min_y; square_y < min_y + (2 * range + 1); ++square_y)
				{
					const int corrected_x = square_x >= 0
						                        ? (square_x < static_cast<int>(screen_x)
							                           ? square_x
							                           : square_x -
							                           static_cast<int>(screen_x))
						                        : static_cast<int>(screen_x) + square_x;
					const int corrected_y = square_y >= 0
						                        ? (square_y < static_cast<int>(screen_y)
							                           ? square_y
							                           : square_y -
							                           static_cast<int>(screen_y))
						                        : static_cast<int>(screen_y) + square_y;

					const std::size_t index_pixel_square = corrected_y * screen_x + corrected_x;

					if (pixels[index_pixel_square].y == 1.0f)
					{
						livings += 1;
					}
				}
			}

			if (pixels[index_pixel].y == 0.0f && livings >= birth_low && livings <= birth_high)
			{
				pixels[index_pixel] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
			}
			else if (pixels[index_pixel].y == 1.0f && (livings < survive_low || livings > survive_high))
			{
				pixels[index_pixel] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
		}
	}
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

void init()
{
	static std::random_device rng;
	std::uniform_int_distribution<> dist(0, 1);
	for (auto& pixel : pixels)
	{
		pixel.y = dist(rng);
	}
}


int main(const int argc, char* argv[])
{
	init_gl(argc, argv);
	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutKeyboardFunc(process_normal_keys);


	glutMainLoop();

	return EXIT_SUCCESS;
}
