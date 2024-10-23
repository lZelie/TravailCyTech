#include <iostream>
#include <array>
#include <chrono>
#include <iomanip>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr std::size_t N = 2 * 0x400ull * 0x400ull;
constexpr std::size_t M = 0x400;
constexpr std::size_t T = 4000;

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

__global__ void add(const float* a, const float* b, float* c, const std::size_t n)
{
	const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
	{
		c[index] = a[index] + b[index];
	}
}

__global__ void chaotic_series(float* a, const std::size_t n)
{
	for (std::size_t i = 0; i < n - 1; i++)
	{
		a[i + 1] = 4 * a[i] * (1 - a[i]);
	}
}

void chaotic_series_a(float* a, const std::size_t n)
{
	for (std::size_t i = 0; i < n - 1; i++)
	{
		a[i + 1] = 4 * a[i] * (1 - a[i]);
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

void test_computing()
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

	for (std::size_t i = 1; i <= M; i *= 2)
	{
		const cuda_timer cuda_timer;
		cuda_timer.start();
		add <<< (N + i - 1) / i, i >>>(d_a, d_b, d_c, N);
		cuda_timer.stop();
		std::cout << "<<< " << (N + i - 1) / i << ", " << i << " >>> => " << cuda_timer << "\n";
	}

	cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

	const auto cpu_c = add_cpu<N>(a, b);

	std::cout << "Error: " << compare_vectors<N>(c, cpu_c) << "\n";

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void test_bandwidth()
{
	float *d_a, *h_b, *d_c, *h_d;

	constexpr std::size_t size = N * sizeof(float);
	const auto a = random_vector<N>();
	const auto b = random_vector<N>();
	std::vector<float> c(N);

	cudaMalloc(&d_a, size);
	cudaMallocHost(&h_b, size);
	cudaMalloc(&d_c, size);
	cudaMallocHost(&h_d, size);

	const cuda_timer cuda_timer;
	cuda_timer.start();
	cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
	cuda_timer.stop();
	std::cout << "Pageable H => D: Speed = " << cuda_timer << ", Bandwidth: " << 1000 * size / (std::pow(1000, 3) *
		cuda_timer.diff()) << "Gb/s\n";
	cuda_timer.start();
	cudaMemcpy(h_b, b.data(), size, cudaMemcpyHostToDevice);
	cuda_timer.stop();
	std::cout << "Pinned H => D: Speed = " << cuda_timer << ", Bandwidth: " << 1000 * size / (std::pow(1000, 3) *
		cuda_timer.diff()) << "Gb/s\n";

	cuda_timer.start();
	cudaMemcpy(d_c, d_a, size, cudaMemcpyDeviceToDevice);
	cuda_timer.stop();
	std::cout << "Pageable D => D: Speed = " << cuda_timer << ", Bandwidth: " << 2 * 1000 * size / (std::pow(1000, 3) *
		cuda_timer.diff()) << "Gb/s\n";
	cuda_timer.start();
	cudaMemcpy(h_d, h_b, size, cudaMemcpyHostToHost);
	cuda_timer.stop();
	std::cout << "Pinned H => H: Speed = " << cuda_timer << ", Bandwidth: " << 2 * 1000 * size / (std::pow(1000, 3) *
		cuda_timer.diff()) << "Gb/s\n";


	cuda_timer.start();
	add <<< (N + M - 1) / M, M>>>(d_a, h_b, d_c, N);
	cuda_timer.stop();
	std::cout << "<<< " << (N + M - 1) / M << ", " << M << " >>> => " << cuda_timer << "\n";
	cuda_timer.start();
	add <<< (N + M - 1) / M, M>>>(d_a, h_b, h_d, N);
	cuda_timer.stop();
	std::cout << "<<< " << (N + M - 1) / M << ", " << M << " >>> => " << cuda_timer << "\n";

	cuda_timer.start();
	cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
	cuda_timer.stop();
	std::cout << "Pageable D => H: Speed = " << cuda_timer << ", Bandwidth: " << 1000 * size / (std::pow(1000, 3) *
		cuda_timer.diff()) << "Gb/s\n";
	cuda_timer.start();
	cudaMemcpy(c.data(), h_d, size, cudaMemcpyDeviceToHost);
	cuda_timer.stop();
	std::cout << "Pinned D => H: Speed = " << cuda_timer << ", Bandwidth: " << 1000 * size / (std::pow(1000, 3) *
		cuda_timer.diff()) << "Gb/s\n";

	const auto cpu_c = add_cpu<N>(a, b);

	std::cout << "Error: " << compare_vectors<N>(c, cpu_c) << "\n";

	cudaFree(d_a);
	cudaFreeHost(h_b);
	cudaFree(d_c);
	cudaFreeHost(h_d);
}

void test_benchmark_compute_bound()
{
	constexpr std::size_t size = N * sizeof(float);
	std::vector<float> a = random_vector<N>();
	float* d_a;

	cudaMalloc(&d_a, size);

	cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

	const cuda_timer cuda_timer;
	cuda_timer.start();
	chaotic_series <<< N / 512, 512 >>> (d_a, T);
	cudaDeviceSynchronize();
	cuda_timer.stop();

	std::cout << "GPU Time: " << cuda_timer << "\n";
	std::cout << "GPU GFlops = " << 3 * T * (N / (cuda_timer.diff() * 1e03f * 1e03f)) << "\n";

	const auto start = std::chrono::system_clock::now();
	chaotic_series_a(d_a, T);
	const auto stop = std::chrono::system_clock::now();
	const auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	std::cout << "CPU Time: " << diff << "ms \n";
	std::cout << "CPU GFlops = " << 3 * T * (N / (diff * 1e03f * 1e03f)) << "\n";





	cudaFree(d_a);

}

int main(int, char*[])
{
	// test_computing();
	// test_bandwidth();
	test_benchmark_compute_bound();
	return EXIT_SUCCESS;
}
