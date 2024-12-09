#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "camera.h"

// OpenGL Graphics includes
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

constexpr auto screen_x = 1600;
constexpr auto screen_y = 912;
constexpr auto fps_update = 200;
constexpr auto title = "N-Body";

constexpr auto mass_max = 5.0f;

constexpr auto cpu_mode = 1;
constexpr auto gpu_mode = 2;
constexpr auto gpu_shared_mode = 3;

constexpr auto g = 1e-7f;
constexpr auto eps2 = 1e-1f;

__device__ __host__ float4 sub(const float4 a, const float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 1.0f);
}

__device__ __host__ float4 add(const float4 a, const float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 1.0f);
}

__device__ __host__ float4 add(const float4 a, const float x)
{
	return make_float4(a.x + x, a.y + x, a.z + x, 1.0f);
}

__device__ __host__ float4 mul(const float4 a, const float x)
{
	return make_float4(a.x * x, a.y * x, a.z * x, 1.0f);
}

__device__ __host__ float4 mul(const float x, const float4 a)
{
	return make_float4(a.x * x, a.y * x, a.z * x, 1.0f);
}

__device__ __host__ float4 div(const float4 a, const float x)
{
	return make_float4(a.x / x, a.y / x, a.z / x, 1.0f);
}

__device__ __host__ float norm(const float4 a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

#define CHECK_CUDA_ERROR(err) check_cuda_error_d(err, __FILE__, __LINE__)

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

/* Globals */

int mode = cpu_mode;
int frame = 0;
int timebase = 0;

std::vector<float4> pos{};
std::vector<float4> vel{};
std::vector<float> mass{};
int nb_bodies = 1024;

std::vector<float4> random_array_float4(const int n, const float d)
{
	static std::random_device rng;
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> dist2(0.0f, 1.0f);
	std::vector<float4> a(n);
	for (int i = 0; i < n; i++)
	{
		const float x = dist(rng);
		const float y = dist(rng);
		const float z = dist(rng);
		const float r = dist2(rng) / std::sqrt(x * x + y * y + z * z);

		a[i].x = r * d * x;
		a[i].y = r * d * y;
		a[i].z = r * d * z;
		a[i].w = 1.0f; // must be 1.0
	}
	return a;
}

std::vector<float> random_array_float(const int n, const float min, const float max)
{
	std::vector<float> a(n);
	static std::random_device rng;
	std::uniform_real_distribution<float> dist(min, max);

	for (int i = 0; i < n; i++)
	{
		a[i] = dist(rng);
	}
	return a;
}

void init_cpu()
{
	pos = random_array_float4(nb_bodies, 1.0f);
	vel = random_array_float4(nb_bodies, 0.0001f);
	mass = random_array_float(nb_bodies, 1.0f, mass_max);
}


float4 *d_pos1, *d_pos2, *d_vel1, *d_vel2;
float* d_mass;

int step = 1;
int nb_threads;
int nb_blocks;

void init_gpu()
{
	pos = random_array_float4(nb_bodies, 1.0f);
	vel = random_array_float4(nb_bodies, 0.0001f);
	mass = random_array_float(nb_bodies, 1.0f, mass_max);
	mass[0] = 10.0f;

	nb_threads = nb_bodies >= 512 ? 512 : nb_bodies;
	nb_blocks = nb_bodies / nb_threads;

	CHECK_CUDA_ERROR(cudaMalloc(&d_pos1, pos.size() * sizeof(float4)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_pos2, pos.size() * sizeof(float4)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_vel1, vel.size() * sizeof(float4)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_vel2, vel.size() * sizeof(float4)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_mass, mass.size() * sizeof(float)));


	CHECK_CUDA_ERROR(
		cudaMemcpy(d_pos1, pos.data(), pos.size() * sizeof(float4), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(
		cudaMemcpy(d_vel1, vel.data(), vel.size() * sizeof(float4), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(
		cudaMemcpy(d_mass, mass.data(), mass.size() * sizeof(float), cudaMemcpyHostToDevice));

	step = 1;
}

void example_cpu()
{
	auto new_pos = pos;
	auto new_vel = vel;
	for (int i = 0; i < nb_bodies; i++)
	{
		float4 acc{0, 0, 0, 1.0f};
		for (int j = 0; j < nb_bodies; ++j)
		{
			const auto r = sub(pos[j], pos[i]);
			const auto d = std::pow(norm(r), 2) + eps2;
			acc = add(acc, mul(g, div(mul(r, mass[j]), std::sqrt(std::pow(d, 3)))));
		}
		new_pos[i] = add(pos[i], vel[i]);
		new_vel[i] = add(vel[i], acc);
	}
	pos = new_pos;
	vel = new_vel;
}

__global__ void calculate_gpu(const float4* d_i_pos, const float4* d_i_vel, const float* d_mass, const int h_nb_bodies,
                              float4* d_o_pos,
                              float4* d_o_vel)
{
	const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;

	float4 acc{0, 0, 0, 1.0f};
	for (int j = 0; j < h_nb_bodies; ++j)
	{
		const auto r = sub(d_i_pos[j], d_i_pos[i]);
		const auto d = std::pow(norm(r), 2) + eps2;
		acc = add(acc, mul(g, div(mul(r, d_mass[j]), std::sqrt(std::pow(d, 3)))));
	}
	d_o_pos[i] = add(d_i_pos[i], d_i_vel[i]);
	d_o_vel[i] = add(d_i_vel[i], acc);
}

__global__ void calculate_gpu_shared(const float4* d_i_pos, const float4* d_i_vel, const float* d_mass, const int h_nb_bodies, const int block_size,
                              float4* d_o_pos,
                              float4* d_o_vel)
{
	const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bDim = blockDim.x;
    const int i = bid * bDim + tid;

    __shared__ float4 shared_pos[512];
    __shared__ float shared_mass[512];

    float4 acc = {0.0f, 0.0f, 0.0f, 1.0f};

    // Load initial data into shared memory
    shared_pos[tid] = d_i_pos[i];
    shared_mass[tid] = d_mass[i];
    __syncthreads();

    // Process bodies in blocks of BLOCK_SIZE
    for (int offset = 0; offset < h_nb_bodies; offset += block_size) {
        if (i + offset < h_nb_bodies) {
            shared_pos[tid] = d_i_pos[i + offset];
            shared_mass[tid] = d_mass[i + offset];
        }
        __syncthreads();

        for (int j = 0; j < block_size; ++j) {
            if (i != j + offset) {
                const auto r = sub(shared_pos[j], shared_pos[tid]);
                const auto d = std::pow(norm(r), 2) + eps2;
                acc = add(acc, mul(g, div(mul(r, shared_mass[j]), std::sqrt(std::pow(d, 3)))));
            }
        }
        __syncthreads();
    }

    d_o_pos[i] = add(d_i_pos[i], d_i_vel[i]);
    d_o_vel[i] = add(d_i_vel[i], acc);
}


void example_gpu()
{
	if (step == 1)
	{
		calculate_gpu <<< nb_blocks, nb_threads >>>(d_pos1, d_vel1, d_mass, nb_bodies, d_pos2, d_vel2);

		CHECK_CUDA_ERROR(cudaGetLastError());

		CHECK_CUDA_ERROR(
			cudaMemcpy(pos.data(), d_pos2, pos.size() * sizeof(float4), cudaMemcpyDeviceToHost));
		CHECK_CUDA_ERROR(
			cudaMemcpy(vel.data(), d_vel2, vel.size() * sizeof(float4), cudaMemcpyDeviceToHost));

		step = 2;
	}
	else
	{
		calculate_gpu <<< nb_blocks, nb_threads >>>(d_pos2, d_vel2, d_mass, nb_bodies, d_pos1, d_vel1);

		CHECK_CUDA_ERROR(cudaGetLastError());

		CHECK_CUDA_ERROR(
			cudaMemcpy(pos.data(), d_pos1, pos.size() * sizeof(float4), cudaMemcpyDeviceToHost));
		CHECK_CUDA_ERROR(
			cudaMemcpy(vel.data(), d_vel1, vel.size() * sizeof(float4), cudaMemcpyDeviceToHost));

		step = 1;
	}
}

void example_gpu_shared()
{
	if (step == 1)
	{
		calculate_gpu_shared <<< nb_blocks, nb_threads >>>(d_pos1, d_vel1, d_mass, nb_bodies, nb_threads, d_pos2, d_vel2);

		CHECK_CUDA_ERROR(cudaGetLastError());

		CHECK_CUDA_ERROR(
			cudaMemcpy(pos.data(), d_pos2, pos.size() * sizeof(float4), cudaMemcpyDeviceToHost));
		CHECK_CUDA_ERROR(
			cudaMemcpy(vel.data(), d_vel2, vel.size() * sizeof(float4), cudaMemcpyDeviceToHost));

		step = 2;
	}
	else
	{
		calculate_gpu_shared <<< nb_blocks, nb_threads >>>(d_pos2, d_vel2, d_mass, nb_bodies, nb_threads, d_pos1, d_vel1);

		CHECK_CUDA_ERROR(cudaGetLastError());

		CHECK_CUDA_ERROR(
			cudaMemcpy(pos.data(), d_pos1, pos.size() * sizeof(float4), cudaMemcpyDeviceToHost));
		CHECK_CUDA_ERROR(
			cudaMemcpy(vel.data(), d_vel1, vel.size() * sizeof(float4), cudaMemcpyDeviceToHost));

		step = 1;
	}
}

void calc_n_bodies()
{
	frame++;
	const int time_cur = glutGet(GLUT_ELAPSED_TIME);

	if (time_cur - timebase > fps_update)
	{
		auto m = "";
		switch (mode)
		{
		case cpu_mode: m = "CPU";
			break;
		case gpu_mode: m = "GPU";
			break;
		case gpu_shared_mode: m = "GPU SHARED";
			break;
		default: break;
		}
		std::stringstream ss;
		ss << title << " (mode: " << m << ", bodies: " << nb_bodies << ", FPS: " << std::fixed << std::setprecision(10)
			<<
			static_cast<float>(frame) * 1000.0f / static_cast<float>(time_cur - timebase) << ")";
		glutSetWindowTitle(ss.str().c_str());
		timebase = time_cur;
		frame = 0;
	}

	switch (mode)
	{
	case cpu_mode: example_cpu();
		break;
	case gpu_mode: example_gpu();
		break;
	case gpu_shared_mode: example_gpu_shared();
		break;
	default: break;
	}
}

void idle_n_bodies()
{
	glutPostRedisplay();
}


void render_n_bodies()
{
	calc_n_bodies();
	camera_apply();

	glClear(GL_COLOR_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, pos.data());
	glDrawArrays(GL_POINTS, 0, nb_bodies);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

void clean_gpu()
{
	CHECK_CUDA_ERROR(cudaFree(d_pos1));
	CHECK_CUDA_ERROR(cudaFree(d_pos2));
	CHECK_CUDA_ERROR(cudaFree(d_vel1));
	CHECK_CUDA_ERROR(cudaFree(d_vel2));
	CHECK_CUDA_ERROR(cudaFree(d_mass));
}

void clean()
{
	if (mode == gpu_mode)
	{
		clean_gpu();
	}
}

void init()
{
	switch (mode)
	{
	case cpu_mode: init_cpu();
		break;
	case gpu_mode:
	case gpu_shared_mode: init_gpu();
		break;
	default: break;
	}
}

void toggle_mode(const int m)
{
	clean();
	mode = m;
	init();
}

void process_normal_keys(const unsigned char key, int, int)
{
	if (key == 27) std::exit(0);
	else if (key == '1') toggle_mode(cpu_mode);
	else if (key == '2') toggle_mode(gpu_mode);
	else if (key == '3') toggle_mode(gpu_shared_mode);
}

void process_special_keys(const int key, int, int)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		if (nb_bodies < 16) nb_bodies++;
		else if (nb_bodies < 128) nb_bodies += 16;
		else if (nb_bodies < 1024) nb_bodies += 128;
		else nb_bodies += 512;
		toggle_mode(mode);
		break;
	case GLUT_KEY_DOWN:
		if (nb_bodies > 1024) nb_bodies -= 512;
		else if (nb_bodies > 128) nb_bodies -= 128;
		else if (nb_bodies > 16) nb_bodies -= 16;
		else if (nb_bodies > 1) nb_bodies--;
		toggle_mode(mode);
		break;
	default: break;
	}
}

void init_gl(int argc, char** argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(screen_x, screen_y);
	glutCreateWindow(title);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glColor4f(1.0, 1.0, 1.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	glPointSize(2.0f);
}


int main(const int argc, char** argv)
{
	init_gl(argc, argv);


	toggle_mode(cpu_mode);

	glutDisplayFunc(render_n_bodies);
	glutIdleFunc(idle_n_bodies);
	glutMouseFunc(trackballMouseFunction);
	glutMotionFunc(trackballMotionFunction);
	glutKeyboardFunc(process_normal_keys);
	glutSpecialFunc(process_special_keys);

	glutMainLoop();

	clean();

	return 1;
}
