#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define MEASURE_CUDA
// #define DO_MULTIPLE_ADD_PER_THREAD
// #define LOCATE

const int IT_NUM = 10, MAX_OP = 1024, SIZE = 64 * MAX_OP, OP_NUM = SIZE/MAX_OP;

void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

double starttime, elapsedtime;

int main()
{
	const int vectorSize = SIZE;
	int a[vectorSize], b[vectorSize], c[vectorSize];
	
	fillVector(a, vectorSize);
	fillVector(b, vectorSize);
	
	
#ifdef MEASURE_CUDA
	addWithCuda(c, a, b, vectorSize);
#else
	starttime = omp_get_wtime();
	for(int _=0 ; _<IT_NUM ; _++)
		addVector(a, b, c, vectorSize);
	elapsedtime = omp_get_wtime() - starttime;
#endif
 	printf("Time Elapsed: %f Secs\n", elapsedtime / IT_NUM);

	// printVector(c, vectorSize);

	return EXIT_SUCCESS;
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}

void ensure(cudaError_t cudaStatus, const char *msg) {
	if(cudaStatus != cudaSuccess) {
		puts(msg);
		exit(1);
	}
}

__global__ void addMultipleOpKernel(int *c, const int *a, const int *b) {
	int l = threadIdx.x * OP_NUM, r = l + OP_NUM;
	for(int i=l ; i<r ; i++)
		c[i] = a[i] + b[i];
}

__global__ void addKernel(int *c, const int *a, const int *b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
#ifdef LOCATE
	printf("Calculated Thread: %d - Block: %d - Warp: %d - Thread %d\n",
		i,
		blockIdx.x,
		threadIdx.x / warpSize,
		threadIdx.x
	);
#endif
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = NULL;
	int *dev_b = NULL;
	int *dev_c = NULL;
	cudaError_t cudaStatus;	

	ensure(cudaSetDevice(0), "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");

	ensure(cudaMalloc((void**)&dev_c, size * sizeof(int)), "cudaMalloc failed!");
	ensure(cudaMalloc((void**)&dev_a, size * sizeof(int)), "cudaMalloc failed!");
	ensure(cudaMalloc((void**)&dev_b, size * sizeof(int)), "cudaMalloc failed!");

	ensure(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
	ensure(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");

	starttime = omp_get_wtime();
	for(int _=0 ; _<IT_NUM ; _++)
#ifdef DO_MULTIPLE_ADD_PER_THREAD
		addMultipleOpKernel <<<1, MAX_OP>>>(dev_c, dev_a, dev_b);
#else
		addKernel <<<SIZE/MAX_OP, MAX_OP>>>(dev_c, dev_a, dev_b);
#endif
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}

	ensure(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code %d after launching addKernel!\n");

	elapsedtime = omp_get_wtime() - starttime;

	ensure(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaSuccess;
}

/*
	serial
	524288: 0.001701

	cuda single op
	524288: 0.000115

	cuda multiple op
	524288: 0.001813
*/
