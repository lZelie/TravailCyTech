#include <array>
#include <assert.h>
#include <corecrt_math.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <GL/freeglut_std.h>

#define CHECK_CUDA_ERROR(err) check_cuda_error_d(err, __FILE__, __LINE__)

struct sphere
{
	float3 rgb;
	float radius;
	float3 xyz;

	__host__ __device__ float hit(float cx, float cy, float* sh) const;
};

enum class render_mode: std::uint8_t
{
	cpu,
	gpu,
	gpu_2,
	gpu_3,
};

constexpr std::size_t screen_x = 1024;
constexpr std::size_t screen_y = 768;
constexpr std::size_t fps_update = 500;
constexpr auto title = "RayTracer";

std::vector<sphere> spheres{};
std::array<float4, screen_x * screen_y> pixels{};
std::size_t frame = 0;
std::size_t time_base = 0;
float scale = 3e-3f;
__device__ float ambient_light = 0.2f;

float camera_x = 0.0f;
float camera_y = 0.0f;

auto render_mode = render_mode::gpu;

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

__global__ void ray_trace_gpu(const sphere* spheres, const std::size_t size_sphere, const std::size_t n_x,
                              const std::size_t n_y, const float scale, float4* pixels)
{
	const std::size_t index_x = threadIdx.x + blockIdx.x * blockDim.x;
	const std::size_t index_y = threadIdx.y + blockIdx.y * blockDim.y;
	const std::size_t index_pixel = index_y * screen_x + index_x;
	if (index_x < n_x && index_y < n_y && size_sphere != 0)
	{
		const float x = scale * (static_cast<float>(index_x) - static_cast<float>(screen_x) / 2.0f);
		const float y = scale * (static_cast<float>(index_y) - static_cast<float>(screen_y) / 2.0f);
		float last_d = -1;
		pixels[index_pixel] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		for (const sphere* sphere = spheres; sphere != spheres + size_sphere; sphere++)
		{
			float sh;
			const float d = sphere->hit(x, y, &sh);
			if (d > last_d && d > 0)
			{
				pixels[index_pixel] = make_float4(sphere->rgb.x * sh, sphere->rgb.y * sh, sphere->rgb.z * sh, 1.0f);
				last_d = d;
			}
		}
	}
}

void ray_trace_cpu(const std::vector<sphere>& spheres)
{
	for (std::size_t index_x = 0; index_x < screen_x; index_x++)
	{
		for (std::size_t index_y = 0; index_y < screen_y; index_y++)
		{
			const std::size_t index_pixel = index_y * screen_x + index_x;
			const float x = scale * (static_cast<float>(index_x) - static_cast<float>(screen_x) / 2.0f);
			const float y = scale * (static_cast<float>(index_y) - static_cast<float>(screen_y) / 2.0f);
			float last_d = -1;
			assert(pixels.size() != 0);
			pixels[index_pixel] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

			for (const sphere& sphere : spheres)
			{
				float sh;
				const float d = sphere.hit(x, y, &sh);
				if (d > last_d && d > 0)
				{
					pixels[index_pixel] = make_float4(sphere.rgb.x * sh, sphere.rgb.y * sh, sphere.rgb.z * sh,
					                                  1.0f);
					last_d = d;
				}
			}
		}
	}
}

float sphere::hit(const float cx, const float cy, float* sh) const
{
	const float dx = cx - xyz.x;
	const float dy = cy - xyz.y;
	const float dz2 = radius * radius - dx * dx - dy * dy;
	if (dz2 > 0)
	{
		const float dz = sqrtf(dz2);
		*sh = dz / radius;
		*sh = ambient_light + *sh * (1 - ambient_light);
		return dz * xyz.z;
	}
	return -INFINITY;
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
	float4* d_pixels;
	sphere* d_spheres;

	CHECK_CUDA_ERROR(cudaMalloc(&d_pixels, pixels.size() * sizeof(float4)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_spheres, spheres.capacity() * sizeof(sphere)));

	CHECK_CUDA_ERROR(
		cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(sphere), cudaMemcpyHostToDevice));

	ray_trace_gpu <<< dim3(screen_x / 16, screen_y / 16), dim3(16, 16) >>>(
		d_spheres, spheres.size(), screen_x, screen_y, scale, d_pixels);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());

	CHECK_CUDA_ERROR(
		cudaMemcpy(pixels.data(), d_pixels, pixels.size() * sizeof(float4), cudaMemcpyDeviceToHost));

	CHECK_CUDA_ERROR(cudaFree(d_spheres));
	CHECK_CUDA_ERROR(cudaFree(d_pixels));
}


void calculate_pixels_cpu()
{
	ray_trace_cpu(spheres);
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

	const float dx = camera_x;
	const float dy = camera_y;
	camera_x = camera_y = 0.0f;

	for (auto& sphere : spheres)
	{
		sphere.xyz.x += dx;
		sphere.xyz.y += dy;
	}

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
	case '3':
		render_mode = render_mode::gpu_2;
		glutPostRedisplay();
		break;
	case '4':
		render_mode = render_mode::gpu_3;
		glutPostRedisplay();
		break;
	case '+':
		{
			sphere add{random_float3(0.2f, 1.0f), random_float(0.2f, 0.6f), random_float3(-10.0f, 10.0f)};
			add.xyz.z = random_float(0.0f, 50.0f);
			spheres.push_back(add);
			glutPostRedisplay();
			break;
		}
	default:
		break;
	}
}

void init()
{
	sphere add{random_float3(0.2f, 1.0f), random_float(0.2f, 0.6f), random_float3(-10.0f, 10.0f)};
	add.xyz.z = random_float(0.0f, 50.0f);
	spheres.push_back(add);
}

bool dragging;

void mouse_motion(int x, int y)
{
	x -= screen_x / 2;
	y -= screen_y / 2;
	static int last_x, last_y;

	if (dragging)
	{
		constexpr float sensitivity = 2e-2f;
		camera_x -= static_cast<float>(last_x - x) * sensitivity;
		camera_y += static_cast<float>(last_y - y) * sensitivity;
	}

	last_x = x;
	last_y = y;

	glutPostRedisplay();
}

void mouse_button(const int button, const int state, const int x, const int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		dragging = state == GLUT_DOWN;
	}

	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

int main(const int argc, char* argv[])
{
	init_gl(argc, argv);
	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutKeyboardFunc(process_normal_keys);

	glutMotionFunc(mouse_motion);
	glutMouseFunc(mouse_button);

	glutMainLoop();

	return EXIT_SUCCESS;
}
