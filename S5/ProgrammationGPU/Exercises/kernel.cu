#include <iostream>
#include <array>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr std::size_t N = 90000000;
constexpr std::size_t M = 512;

__global__ void add(const float* a, const float* b, float* c, const std::size_t n)
{
	const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
	{
		c[index] = a[index] + b[index];
	}
}

template <std::size_t Size>
std::vector<float> add_cpu(const std::vector<float>& a, const std::vector<float>& b)
{
	std::vector<float> ret(Size);
	for (std::size_t i = 0; i < Size; i++)
	{
		ret[i] = a[i] + b[i];
	}
	return ret;
}

template <std::size_t Size>
float compare_vectors(const std::vector<float>& a, const std::vector<float>& b)
{
	float diff = 0.0f;
	for (std::size_t i = 0; i < Size; i++)
	{
		diff += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return std::sqrt(diff / Size);
}

template <std::size_t Size>
std::vector<float> random_vector()
{
	static std::random_device rng;
	std::vector<float> ret(Size);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (auto& f : ret)
	{
		f = dist(rng);
	}
	return ret;
}

int main(int, char*[])
{
	float *d_a, *d_b, *d_c;

	constexpr std::size_t size = N * sizeof(float);
	const auto a = random_vector<N>();
	const auto b = random_vector<N>();
	std::vector<float> c(N);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);
	cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
	add <<< (N + M - 1) / M, M >>>(d_a, d_b, d_c, N);
	cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

	const auto cpu_c = add_cpu<N>(a, b);

	std::cout << "Error: " << compare_vectors<N>(c, cpu_c) << "\n";

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return EXIT_SUCCESS;
}
