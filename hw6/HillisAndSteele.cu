#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "config.h"

void fill_array(DTYPE *a, size_t n);
void prefix_sum(DTYPE *a, size_t n);
void print_array(DTYPE *a, size_t n);
void ensure(cudaError_t error, const char *msg);

int main(int argc, char *argv[]) {
	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);
	// Allocate memory for array
	DTYPE * a = (DTYPE *)malloc(n * sizeof a);
	// Fill array with numbers 1..n
	fill_array(a, n);
	// Print array
	// print_array(a, n);
	// Compute prefix sum
	prefix_sum(a, n);
	// Print array
	// print_array(a, n);
	// Free allocated memory
	free(a);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    // printf(deviceProp.)
	return EXIT_SUCCESS;
}

void ensure(cudaError_t error, const char *msg) {
    if(error != cudaSuccess) {
        fprintf(stderr, msg);
        fprintf(stderr, " ---- error code: %s\n", cudaGetErrorString(error));
        
        exit(EXIT_FAILURE);
    }
}

__global__ void 
HAS(DTYPE *d_a, DTYPE *tmp, size_t n, int step, int d) {
    DTYPE *a[2] = {d_a, tmp};

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n)
        return;
    a[d][id] = a[1-d][id];
    if (id >= step)
        a[d][id] += a[1-d][id - step];
       
}

void prefix_sum(DTYPE *a, size_t n) {
    size_t size = n * sizeof(DTYPE);
    DTYPE *d_a, *tmp;

    ensure(cudaMalloc((void**)&d_a, size), "could not allocate d_a in device.");
    ensure(cudaMalloc((void**)&tmp, size), "could not allocate tmp in device.");
    
    ensure(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice), "could not copy to device");

    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    cudaEvent_t start, end;
    ensure(cudaEventCreate(&start), "could not create event for start.");
    ensure(cudaEventCreate(&end), "could not create event for end.");

    ensure(cudaEventRecord(start, NULL), "could not record start.");
    
    int d=1;
    for(int step=1; step<n ; step <<= 1, d = 1-d)
        HAS <<< grid, block >>> (d_a, tmp, n, step, d);
    
    cudaDeviceSynchronize();

    ensure(cudaEventRecord(end, NULL), "could not record end.");
    ensure(cudaEventSynchronize(end), "could not sync.");

    float elapsedtime;
    ensure(cudaEventElapsedTime(&elapsedtime, start, end), "could not calc elapsed time.");

    printf("Elapsed time in msec = %f\n", elapsedtime);

    if(d)
        ensure(cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost), "could not copy the result");
    else
        ensure(cudaMemcpy(a, tmp, size, cudaMemcpyDeviceToHost), "could not copy the result");
    
    ensure(cudaFree(d_a), "could not free d_a from device.");
    ensure(cudaFree(tmp), "could not free tmp from device.");
}

void print_array(DTYPE *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf(DTYPE_FORMAT, a[i]);
        printf(", ");
	}
	printf("\n");
}

void fill_array(DTYPE *a, size_t n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}