#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


constexpr std::size_t tile_width = 0x40;
constexpr std::size_t width = 0x200;

class cuda_timer
{
private:
	cudaEvent_t cstart;
	cudaEvent_t cstop;

public:
	cuda_timer()
	{
		cudaEventCreate(&cstart);
		cudaEventCreate(&cstop);
	}

	~cuda_timer()
	{
		cudaEventDestroy(cstop);
		cudaEventDestroy(cstart);
	}

	void start() const
	{
		cudaEventRecord(cstart);
	}

	void stop() const
	{
		cudaEventRecord(cstop);
		cudaEventSynchronize(cstop);
	}

	float diff() const
	{
		float diff;
		cudaEventElapsedTime(&diff, cstart, cstop);
		return diff;
	}

	friend std::ostream& operator<<(std::ostream& os, const cuda_timer& cuda_timer)
	{
		return os << "CUDA time is " << std::fixed << std::setprecision(10) << cuda_timer.diff() << "ms" <<
			std::defaultfloat;
	}
};

__global__ void matrix_mul(const float* m_d, const float* n_d, float* p_d, const std::size_t width)
{
	__shared__ float m_d_s[tile_width][tile_width];
	__shared__ float n_d_s[tile_width][tile_width];
	const std::size_t b_x = blockIdx.x;
	const std::size_t b_y = blockIdx.y;
	const std::size_t t_x = threadIdx.x;
	const std::size_t t_y = threadIdx.y;

	const std::size_t row = b_y * tile_width + t_y;
	const std::size_t col = b_x * tile_width + t_x;

	float p_value = 0;
	for (std::size_t m = 0; m < width / tile_width; ++m)
	{
		m_d_s[t_y][t_x] = m_d[row * width + (m * tile_width + t_x)];
		n_d_s[t_y][t_x] = n_d[col + (m * tile_width + t_x) * width];

		__syncthreads();

		for (int k = 0; k < tile_width; ++k)
		{
			p_value += m_d_s[t_y][k] * n_d_s[k][t_x];
			__syncthreads();
		}
		p_d[row * width + col] = p_value;
	}
}

std::vector<float> cpu_matrix_mul(const std::vector<float>& m, const std::vector<float>& n, const std::size_t width)
{
	std::vector<float> p(width * width);
	for (std::size_t i = 0; i < width; ++i)
	{
		for (std::size_t j = 0; j < width; ++j)
		{
			float sum = 0;
			for (int k = 0; k < width; ++k)
			{
				const float a = m[i * width + k];
				const float b = m[j + width * k];
				sum += a * b;
			}
			p[i * width + j] = sum;
		}
	}
	return p;
}


std::vector<float> random_vector(const std::size_t vector_size)
{
	static std::random_device rng;
	std::vector<float> ret(vector_size);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (auto& f : ret)
	{
		f = dist(rng);
	}
	return ret;
}

int main(int argc, char* argv[])
{
	const auto matrix_m = random_vector(width * width);
	const auto matrix_n = random_vector(width * width);

	float *m_d, *n_d, *p_d;

	cudaMalloc(&m_d, width * width * sizeof(float));
	cudaMalloc(&n_d, width * width * sizeof(float));
	cudaMalloc(&p_d, width * width * sizeof(float));

	cudaMemcpy(m_d, matrix_m.data(), width * width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(n_d, matrix_n.data(), width * width * sizeof(float), cudaMemcpyHostToDevice);

	const cuda_timer cuda_timer;
	cuda_timer.start();
	matrix_mul <<< dim3(width / tile_width, width / tile_width), dim3(tile_width, tile_width) >>>(m_d, n_d, p_d, width);
	cuda_timer.stop();

	std::cout << "GPU Matrix multiplication: " << cuda_timer << std::endl;

	cuda_timer.start();
	cpu_matrix_mul(matrix_m, matrix_n, width);
	cuda_timer.stop();

	std::cout << "CPU Matrix multiplication: " << cuda_timer << std::endl;

	cudaFree(m_d);
	cudaFree(n_d);
	cudaFree(p_d);
}
