#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


constexpr std::size_t size = 0x1000000;

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

__constant__ float d_const[4096];

__global__ void test_global_memory(const float* d_in_data, float* d_odata)
{
	const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 0;
	for (const float* j = d_in_data; j != d_in_data + size; j++) x += *j;
	d_odata[i] = x;
}

__global__ void test_const_memory(float* d_odata)
{
	const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 0;
	for (const float j : d_const) x += j;
	d_odata[i] = x;
}

__global__ void test_register(const float f, float* d_odata)
{
	const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 0;
	for (std::size_t j = 0; j < size; j++) x += f;
	d_odata[i] = x;
}

constexpr std::size_t data_size = 32 * 1024 * 1024;
constexpr std::size_t nb_threads = 512;
constexpr std::size_t n = data_size / nb_threads;

__global__ void dot(const float* a, const float* b, float* c)
{
	__shared__ float sum_tab[nb_threads];
	sum_tab[threadIdx.x] = 0;
	const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	for (std::size_t i = index; i < n; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
	{
		sum_tab[threadIdx.x] += a[i] * b[i];
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sum = 0;
		for (const float i : sum_tab)
		{
			sum += i;
		}
		atomicAdd(c, sum);
	}
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

/*void memory_test()
{
	constexpr std::size_t data_size = 1024ull * 1024ull;
	constexpr std::size_t mem_size = sizeof(float) * data_size;

	const std::vector<float> in_vector = random_vector(size);

	float* d_in;
	float* d_out;

	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, mem_size);

	cudaMemcpy(d_in, in_vector.data(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_const, in_vector.data(), size * sizeof(float));


	const cuda_timer cuda_timer;
	cuda_timer.start();
	test_global_memory <<< data_size / 512, 512 >>>(d_in, d_out);
	cuda_timer.stop();
	std::cout << "Global memory: " << cuda_timer << "\n";

	cuda_timer.start();
	test_const_memory <<< data_size / 512, 512 >>>(d_out);
	cuda_timer.stop();
	std::cout << "Constant memory: " << cuda_timer << "\n";

	cuda_timer.start();
	test_register <<< data_size / 512, 512 >>>(0.5, d_out);
	cuda_timer.stop();
	std::cout << "Register:  " << cuda_timer << "\n";

	cudaFree(d_in);
	cudaFree(d_out);
}*/

int main(int argc, char* argv[])
{
	const std::vector<float> in_vector_a = random_vector(data_size);
	const std::vector<float> in_vector_b = random_vector(data_size);

	float* d_in_a;
	float* d_in_b;
	float* d_out;

	cudaMalloc(&d_in_a, data_size * sizeof(float));
	cudaMalloc(&d_in_b, data_size * sizeof(float));
	cudaMalloc(&d_out, sizeof(float));

	cudaMemcpy(d_in_a, in_vector_a.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_b, in_vector_b.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);


	for (std::size_t i = 1; i <= n; i *= 2)
	{
		const cuda_timer cuda_timer;
		cuda_timer.start();
		dot <<< i, nb_threads >>>(d_in_a, d_in_b, d_out);
		cuda_timer.stop();
		std::cout << "Dot product(" << i << " blocks, " << (n + 1) / i <<" loops) : " << cuda_timer << "\n";
	}

	cudaFree(d_in_a);
	cudaFree(d_in_b);
	cudaFree(d_out);
}
