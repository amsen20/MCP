/*
*				In His Exalted Name
*	Title:	Pseudo-Timsort Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	24/11/2015
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define MAX(A, B) ((A)>(B))?(A):(B)
#define MIN(A, B) ((A)<(B))?(A):(B)

const int TIM_SORT_CUT = 10;
const int IT_NUM = 2;
const int MIN_TASK_SIZE = (1 << 10);

void printArray(int *array, int size);
void fillArray(int *array, int size);
void merge(int *a, int n, int m);
void mergeSort (int *a, int n);
void insertionSort(int *a, int n);
void timSort (int *a, int n);
void ptimSort (int *a, int n);
void run_ptimSort(int *a, int n);


int main(int argc, char *argv[]){
	srand(time(NULL));
	int *array = NULL;
	int size = atoi(argv[1]);
	int thread_num = atoi(argv[2]);
	omp_set_num_threads(thread_num);

	array = (int *) malloc(sizeof(int) * size);
	double starttime = omp_get_wtime();

	for(int _=0 ; _<IT_NUM ; _++) {
	
		fillArray(array, size);
		run_ptimSort(array, size);
		// timSort(array, size);
		// printArray(array, size);
	
	}

	free(array);
	printf("Time Elapsed %f Secs\n", (omp_get_wtime() - starttime)/IT_NUM);
	return EXIT_SUCCESS;
}

void fillArray(int *array, int size){
	while(size-->0){
		*array++ = rand() % 100;
	}
}

void printArray(int *array, int size){
	while(size-->0){
		printf("%d ", *array++);
	}
	printf("\n");
}

void insertionSort(int *a, int n){
    int i, j, temp;
    for (i = 1; i < n; i++) {
        temp = a[i];
        for (j = i; j > 0 && temp < a[j - 1]; j--) {
            a[j] = a[j - 1];
        }
        a[j] = temp;
    }
}

void merge(int *a, int n, int m){
	int i, j, k;
    int *temp = (int *) malloc(n * sizeof (int));
    for (i = 0, j = m, k = 0; k < n; k++) {
        temp[k] = j == n   ? a[i++]
             : i == m      ? a[j++]
             : a[j] < a[i] ? a[j++]
             :               a[i++];
    }
    for (i = 0; i < n; i++) {
        a[i] = temp[i];
    }
    free(temp);
}

void mergeSort (int *a, int n){
    int m;
	if (n < 2)
        return;
    m = n / 2;
    mergeSort(a, m);
    mergeSort(a + m, n - m);
    merge(a, n, m);
}

void timSort(int *a, int n) {
	if(n <= TIM_SORT_CUT) {
		insertionSort(a, n);
		return;
	}
	int m;
	if (n < 2)
        return;
    m = n / 2;
    timSort(a, m);
    timSort(a + m, n - m);
    merge(a, n, m);
}

void run_ptimSort(int *a, int n) {
	#pragma omp parallel
	{
		#pragma omp single
		ptimSort(a, n);
	}
}

void ptimSort(int *a, int n) {
	if(n <= TIM_SORT_CUT) {
		insertionSort(a, n);
		return;
	}
	int m;
	if (n < 2)
        return;
    m = n / 2;
	
	#pragma omp task shared(a) if (m >= MIN_TASK_SIZE)
	ptimSort(a, m);

	#pragma omp task shared(a) if (n - m >= MIN_TASK_SIZE)
	ptimSort(a + m, n - m);

	#pragma omp taskwait
    merge(a, n, m);
}

/*
	serial = {
		25000: 0.006710,
		250000: 0.040305,
		2500000: 0.548716,
		25000000: 7.935376,
		250000000: 86.114275
	}

	parallel = {
		1: {
			25000: 0.006878,
			250000: 0.041450,
			2500000: 0.586518,
			25000000: 5.626355,
			250000000: 93.282690
		},
		2: {
			25000: 0.002310,
			250000: 0.033920,
			2500000: 0.263727,
			25000000: 3.646013,
			250000000: 46.367473
		},
		4: {
			25000: 0.003442,
			250000: 0.031250,
			2500000: 0.204797,
			25000000: 2.523713,
			250000000: 34.434910
		},
		8: {
			25000: 0.003854,
			250000: 0.031932,
			2500000: 0.208942,
			25000000: 2.085773,
			250000000: 24.524215
		}
	}
*/