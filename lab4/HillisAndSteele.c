/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>

const int IT_NUM = 10;

void omp_check();
void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);

int main(int argc, char *argv[]) {
	// Check for correct compilation settings
    omp_set_num_threads(4);
	omp_check();
	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%lud\n", &n);

	double starttime, elapsedtime;
	starttime = omp_get_wtime();
	for(int it=0 ; it<IT_NUM ; it ++) {
		// Allocate memory for array
		int * a = (int *)malloc(n * sizeof a);
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
	}
	elapsedtime = omp_get_wtime() - starttime;
	printf("Time Elapsed: %f Secs\n", elapsedtime/IT_NUM);
	return EXIT_SUCCESS;
}

void prefix_sum(int *a, size_t n) {
    int * tmp = (int *)malloc(n * sizeof tmp);

	for(int step=1 ; step < n ; step <<= 1) {
        memcpy(tmp, a, n * sizeof a);
        #pragma omp parallel for
        for(int i=n-1 ; i>=step ; i--)
            a[i] = tmp[i] + tmp[i - step];
    }
}

void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}

void fill_array(int *a, size_t n) {
    int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}

void omp_check() {
	printf("------------ Info -------------\n");
#ifdef _DEBUG
	printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
	printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
	printf("[-] Platform: x64\n");
#elif _M_IX86 
	printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
	printf("[-] OpenMP is on.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
	printf("[!] OpenMP is off.\n");
	printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
	printf("[-] Maximum threads: %d\n", omp_get_max_threads());
	printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}

/*
1024: 0.000101
65536: 0.007341
1048576: 0.045577
33554432: 2.015626
*/