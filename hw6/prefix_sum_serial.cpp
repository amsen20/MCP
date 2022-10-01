/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS
#define IT_NUM 10

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);

int main(int argc, char *argv[]) {
	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);

	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);

	double starttime, elapsedtime;
	starttime = omp_get_wtime();

	for(int it=0 ; it<IT_NUM ; it++) {
		// Fill array with numbers 1..n
		fill_array(a, n);
		// Print array
		// print_array(a, n);
		// Compute prefix sum

		prefix_sum(a, n);

		// Print array
		// print_array(a, n);
		// Free allocated memory
	}

	elapsedtime = omp_get_wtime() - starttime;
	printf("Time Elapsed: %f msec\n", 1000 * elapsedtime / IT_NUM);

	free(a);
	return EXIT_SUCCESS;
}

void prefix_sum(int *a, size_t n) {
	int i;
	for (i = 1; i < n; ++i) {
		a[i] = a[i] + a[i - 1];
	}
}

void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\n");
}

void fill_array(int *a, size_t n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}
