#include <cstdio>
#include <cmath>

// OpenGL Graphics includes
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <assert.h>
#include <complex>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <iostream>

#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "Julia Fractals"

#define CPU_MODE 1
#define GPU_MODE 2

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

	friend std::ostream& operator<<(std::ostream& os, const cuda_timer& cuda_timer)
	{
		float diff;
		cudaEventElapsedTime(&diff, cuda_timer.cstart, cuda_timer.cstop);
		return os << "CUDA time is " << std::fixed << std::setprecision(2) << diff << "ms\n";
	}
};

GLuint imageTex;
GLuint imageBuffer;
float* debug;

/* Globals */
float scale = 0.003f;
float mx, my;
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;
int precision = 10;

float4* pixels;

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

float julia_color(const float x, const float y, const float sx, const float sy, const std::size_t p)
{
	std::complex<float> a(x, y);
	const std::complex<float> seed(sx, sy);
	for (std::size_t i = 0; i < p; i++)
	{
		a = a * a + seed;
		if (std::pow(std::abs(a), 2) > 4)
		{
			return 1 - static_cast<float>(i) / static_cast<float>(p);
		}
	}
	return 0;
}

__device__ float d_julia_color(const float x, const float y, const float sx, const float sy, const std::size_t p)
{
	float a_r = x;
	float a_i = y;
	const float seed_r = sx;
	const float seed_i = sy;
	for (std::size_t i = 0; i < p; i++)
	{
		const float temp_r = a_r * a_r - a_i * a_i + seed_r;
		a_i = a_r * a_i + a_i * a_r + seed_i;
		a_r = temp_r;
		const float magnitude = sqrtf(powf(a_r, 2) + powf(a_i, 2));
		if (powf(magnitude, 2) > 4)
		{
			return 1 - static_cast<float>(i) / static_cast<float>(p);
		}
	}
	return 0;
}

__global__ void gpu_julia_color(const float sx, const float sy, const std::size_t p, const std::size_t n_x,
                                const std::size_t n_y, const float scale, float4* pixels)
{
	const std::size_t index_x = threadIdx.x + blockIdx.x * blockDim.x;
	const std::size_t index_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (index_x < n_x && index_y < n_y)
	{
		const float x = scale * (static_cast<float>(index_x) - static_cast<float>(SCREEN_X) / 2.0f);
		const float y = scale * (static_cast<float>(index_y) - static_cast<float>(SCREEN_Y) / 2.0f);
		const float color = d_julia_color(x, y, sx, sy, p);
		pixels[index_y * SCREEN_X + index_x] = make_float4(color, color, color, 1.0f);
	}
}

void init_cpu()
{
	pixels = (float4*)malloc(SCREEN_X * SCREEN_Y * sizeof(float4));
}

void clean_cpu()
{
	free(pixels);
}

void init_gpu()
{
	pixels = (float4*)malloc(SCREEN_X * SCREEN_Y * sizeof(float4));
}

void clean_gpu()
{
	free(pixels);
}

void example_cpu()
{
	for (int i = 0; i < SCREEN_Y; i++)
		for (int j = 0; j < SCREEN_X; j++)
		{
			float x = (float)(scale * (j - SCREEN_X / 2));
			float y = (float)(scale * (i - SCREEN_Y / 2));
			float4* p = pixels + (i * SCREEN_X + j);

			const float color = julia_color(x, y, mx, my, precision);

			p->x = color;
			p->y = color;
			p->z = color;
			p->w = 1.0f;
		}
}

void example_gpu()
{
	float4* d_pixels;
	CHECK_CUDA_ERROR(cudaMalloc(&d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4)));

	gpu_julia_color <<< dim3(SCREEN_X / 16, SCREEN_Y / 16), dim3(16, 16) >>>(
		mx, my, precision, SCREEN_X, SCREEN_Y, scale, d_pixels);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());

	cudaMemcpy(pixels, d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);
}

void calculate()
{
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE)
	{
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU mode";
			break;
		case GPU_MODE: m = "GPU mode";
			break;
		}
		sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: example_cpu();
		break;
	case GPU_MODE: example_gpu();
		break;
	}
}

void idle()
{
	glutPostRedisplay();
}


void render()
{
	calculate();
	switch (mode)
	{
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels);
		break;
	case GPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels);
		break;
	}
	glutSwapBuffers();
}

void clean()
{
	switch (mode)
	{
	case CPU_MODE: clean_cpu();
		break;
	case GPU_MODE: clean_gpu();
		break;
	}
}

void init()
{
	switch (mode)
	{
	case CPU_MODE: init_cpu();
		break;
	case GPU_MODE: init_gpu();
		break;
	}
}

void toggle_mode(int m)
{
	clean();
	mode = m;
	init();
}

void mouse(int button, int state, int x, int y)
{
	if (button <= 2)
	{
		mx = (float)(scale * (x - SCREEN_X / 2));
		my = -(float)(scale * (y - SCREEN_Y / 2));
	}
	// Wheel reports as button 3 (scroll up) and button 4 (scroll down)
	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y)
{
	mx = (float)(scale * (x - SCREEN_X / 2));
	my = -(float)(scale * (y - SCREEN_Y / 2));
}

void process_normal_keys(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		clean();
		exit(0);
	case '1':
		toggle_mode(CPU_MODE);
		break;
	case '2':
		toggle_mode(GPU_MODE);
		break;
	case '+':
		precision *= 2;
		glutPostRedisplay();
		break;
	case '-':
		precision /= 2;
		glutPostRedisplay();
		break;
	}
}

void process_special_keys(int key, int x, int y)
{
	// other keys (F1, F2, arrows, home, etc.)
	switch (key)
	{
	case GLUT_KEY_UP: break;
	case GLUT_KEY_DOWN: break;
	}
}

void init_gl(int argc, char** argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_X, SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glDisable(GL_DEPTH_TEST);

	// View Ortho
	// Sets up the OpenGL window so that (0,0) corresponds to the top left corner, 
	// and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner.  
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_X, SCREEN_Y, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}


int main(int argc, char** argv)
{
	init_gl(argc, argv);

	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(process_normal_keys);
	glutSpecialFunc(process_special_keys);

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}
