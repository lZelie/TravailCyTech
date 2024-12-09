#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <numeric>

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

constexpr std::size_t N = 64 * 0x400ull * 0x400ull;
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
        return os << "time is " << std::fixed << std::setprecision(10) << cuda_timer.diff() << "ms" <<
            std::defaultfloat;
    }
};

__global__ void kernel_warp_ok(const float* d_i_data, float* d_o_data)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = d_i_data[i];
    if (threadIdx.x / 32 % 2 == 0)
    {
        for (std::size_t j = 0; j < 100; ++j) x += .1;
        d_o_data[i] = x;
    }
    else
    {
        for (std::size_t j = 0; j < 100; ++j) x += .2;
        d_o_data[i] = x;
    }
}

__global__ void kernel_warp_ko(const float* d_i_data, float* d_o_data)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = d_i_data[i];
    if (threadIdx.x % 2 == 0)
    {
        for (std::size_t j = 0; j < 100; ++j) x += .1;
        d_o_data[i] = x;
    }
    else
    {
        for (std::size_t j = 0; j < 100; ++j) x += .2;
        d_o_data[i] = x;
    }
}

__global__ void coalesce(const float* d_i_data, float* d_o_data, const int data_size, const int stride)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = stride * i % data_size;
    d_o_data[j] = d_i_data[i];
}

__global__ void bank(const float* d_i_data, float* d_o_data, const int stride)
{
    __shared__ double sm[512];
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    sm[threadIdx.x] = d_i_data[i];
    __syncthreads();
    const std::size_t n = stride * threadIdx.x % 512;
    d_o_data[i] = sm[n];
}

__global__ void reduce(const int* g_i_data, int* g_o_data)
{
    volatile extern __shared__ int s_data[];

    const std::size_t tid = threadIdx.x;
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[tid] = g_i_data[i];
    __syncthreads();
    for (std::size_t s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_o_data[blockIdx.x] = s_data[0];
    }
}

__global__ void reduce_better(const int* g_i_data, int* g_o_data)
{
    volatile extern __shared__ int s_data[];
    const std::size_t tid = threadIdx.x;
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[tid] = g_i_data[i];
    __syncthreads();
    for (std::size_t s = 1; s < blockDim.x; s *= 2)
    {
        const std::size_t index = 2 * s * tid;
        if (index < blockDim.x)
        {
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_o_data[blockIdx.x] = s_data[0];
    }
}

__global__ void reduce_better_better(const int* g_i_data, int* g_o_data)
{
    volatile extern __shared__ int s_data[];
    const std::size_t tid = threadIdx.x;
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[tid] = g_i_data[i];
    __syncthreads();
    for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_o_data[blockIdx.x] = s_data[0];
    }
}

__global__ void reduce_better_better_better(const int* g_i_data, int* g_o_data)
{
    volatile extern __shared__ int s_data[];
    const std::size_t tid = threadIdx.x;
    const std::size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    s_data[tid] = g_i_data[i] + g_i_data[i + blockDim.x];
    __syncthreads();
    for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_o_data[blockIdx.x] = s_data[0];
    }
}

__global__ void reduce_better_better_better_better(const int* g_i_data, int* g_o_data)
{
	volatile extern __shared__ int s_data[];
    const std::size_t tid = threadIdx.x;
    const std::size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    s_data[tid] = g_i_data[i] + g_i_data[i + blockDim.x];
    __syncthreads();
    for (std::size_t s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
	    for (int s = 32; s > 0; s /= 2)
	    {
		    s_data[tid] += s_data[tid + 2];
	    }
    }
    if (tid == 0)
    {
        g_o_data[blockIdx.x] = s_data[0];
    }
}

__global__ void reduce_better_better_better_better_better(const int* g_i_data, int* g_o_data, const std::size_t data_size)
{
	volatile extern __shared__ int s_data[];
    const std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    const std::size_t grid_size = blockDim.x * 2 * gridDim.x;
    s_data[tid] = 0;
	while (i < data_size)
	{
		s_data[tid] = g_i_data[i] + g_i_data[i + blockDim.x];
        i += grid_size;
	}
    __syncthreads();

    for (std::size_t s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
	    for (int s = 32; s > 0; s /= 2)
	    {
		    s_data[tid] += s_data[tid + 2];
	    }
    }
    if (tid == 0)
    {
        g_o_data[blockIdx.x] = s_data[0];
    }
}

int reduce_cpu(const std::vector<int>& i_data)
{
    return std::accumulate(i_data.begin(), i_data.end(), 0, [](const int a, const int b){return a + b;});
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

void test_warp()
{
    float *d_a, *d_c;

    constexpr std::size_t size = N * sizeof(float);
    const auto a = random_vector(N);
    std::vector<float> c(N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

    const cuda_timer timer;
    timer.start();
    kernel_warp_ok <<< N / 512, 512 >>>(d_a, d_c);
    timer.stop();

    std::cout << "kernel_warp_ok: " << timer << "\n";

    timer.start();
    kernel_warp_ko <<< N / 512, 512 >>>(d_a, d_c);
    timer.stop();

    std::cout << "kernel_warp_ko: " << timer << "\n";

    cudaFree(d_a);
    cudaFree(d_c);
}

void test_coalesce()
{
    float *d_a, *d_c;

    constexpr std::size_t size = N * sizeof(float);
    const auto a = random_vector(N);
    std::vector<float> c(N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

    for (std::size_t i = 0; i < 50; ++i)
    {
        const cuda_timer timer;
        timer.start();
        coalesce <<< N / 512, 512 >>>(d_a, d_c, N, i);
        timer.stop();

        std::cout << timer.diff() << "\n";
    }


    cudaFree(d_a);
    cudaFree(d_c);
}

void test_bank()
{
    float *d_a, *d_c;

    constexpr std::size_t size = N * sizeof(float);
    const auto a = random_vector(N);
    std::vector<float> c(N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

    for (std::size_t i = 0; i < 50; ++i)
    {
        const cuda_timer timer;
        timer.start();
        bank <<< N / 512, 512 >>>(d_a, d_c, i);
        timer.stop();

        std::cout << timer.diff() << "\n";
    }


    cudaFree(d_a);
    cudaFree(d_c);
}

void test_reduce()
{
    int *d_a, *d_c;

    constexpr int n = 512;
    constexpr std::size_t size = n * sizeof(int);
    const std::vector<int> a(n, 2);
    std::vector<int> c(n);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

    const cuda_timer timer;
    timer.start();
    const auto res = reduce_cpu(a);
    timer.stop();

    std::cout << "Cpu: " << timer << ", result: " << res << '\n';

    timer.start();
    reduce <<< n / 512, 512, 512 * sizeof(int) >>>(d_a, d_c);
    timer.stop();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Gpu: " << timer << ", result: " << c[0] << '\n';

    timer.start();
    reduce_better <<< n / 512, 512, 512 * sizeof(int) >>>(d_a, d_c);
    timer.stop();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Gpu better: " << timer << ", result: " << c[0] << '\n';

    timer.start();
    reduce_better_better <<< n / 512, 512, 512 * sizeof(int) >>>(d_a, d_c);
    timer.stop();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Gpu super better: " << timer << ", result: " << c[0] << '\n';

    timer.start();
    reduce_better_better_better <<< n / 512 / 2, 512, 512 * sizeof(int) >>>(d_a, d_c);
    timer.stop();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Gpu super duper better: " << timer << ", result: " << c[0] << '\n';

    timer.start();
    reduce_better_better_better_better <<< n / 512 / 2, 512, 512 * sizeof(int) >>>(d_a, d_c);
    timer.stop();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Gpu super dee duper better: " << timer << ", result: " << c[0] << '\n';

    timer.start();
    reduce_better_better_better_better_better <<< n / 512 / 2, 512, 512 * sizeof(int) >>>(d_a, d_c, n);
    timer.stop();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Gpu supercalifragilisticexpialidocious: " << timer << ", result: " << c[0] << '\n';


    cudaFree(d_a);
    cudaFree(d_c);
}

int main()
{
    /*test_warp();*/
    /*test_coalesce();*/
    /*test_bank();*/
    test_reduce();
}
