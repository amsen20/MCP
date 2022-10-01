#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

typedef struct {
	int *A, *B, *C;
	int n, m, p;
} DataSet;

void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void multiply(DataSet dataSet);
void multiply1D(DataSet dataSet);
void multiply2D(DataSet dataSet);

int main(int argc, char *argv[]) {
	DataSet dataSet;
	if (argc < 4) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> <p>\n");
		printf(">>> ");
		scanf("%d %d %d", &dataSet.n, &dataSet.m, &dataSet.p);
	}
	else {
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
		dataSet.p = atoi(argv[3]);
	}
	double start_time, end_time;
	omp_set_num_threads(8);
	fillDataSet(&dataSet);
	
	start_time = omp_get_wtime();

	// multiply(dataSet);
	// multiply1D(dataSet);
	multiply2D(dataSet);
	end_time = omp_get_wtime();
	printf("time: %lf\n", end_time - start_time);
	// printDataSet(dataSet);
	closeDataSet(dataSet);
	printf("---------------------\n");
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet) {
	int i, j;

	dataSet->A = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->B = (int *)malloc(sizeof(int) * dataSet->m * dataSet->p);
	dataSet->C = (int *)malloc(sizeof(int) * dataSet->n * dataSet->p);

	// srand(time(NULL));

	for (i = 0; i < dataSet->n; i++) {
		for (j = 0; j < dataSet->m; j++) {
			dataSet->A[i*dataSet->m + j] = rand() % 100;
		}
	}

	for (i = 0; i < dataSet->m; i++) {
		for (j = 0; j < dataSet->p; j++) {
			dataSet->B[i*dataSet->p + j] = rand() % 100;
		}
	}

}

void printDataSet(DataSet dataSet) {
	int i, j;

	printf("[-] Matrix A\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.A[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix B\n");
	for (i = 0; i < dataSet.m; i++) {
		for (j = 0; j < dataSet.p; j++) {
			printf("%-4d", dataSet.B[i*dataSet.p + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.p; j++) {
			printf("%-8d", dataSet.C[i*dataSet.p + j]);
		}
		putchar('\n');
	}
}

void closeDataSet(DataSet dataSet) {
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}

void multiply(DataSet dataSet) {
	int i, j, k, sum;
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.p; j++) {
			sum = 0;
			for (k = 0; k < dataSet.m; k++) {
				sum += dataSet.A[i * dataSet.m + k] * dataSet.B[k * dataSet.p + j];
			}
			dataSet.C[i * dataSet.p + j] = sum;
		}
	}
}

void multiply1D(DataSet dataSet) {
	#pragma omp parallel for
	for (int i = 0; i < dataSet.n; i++) {
		for (int j = 0; j < dataSet.p; j++) {
			int sum = 0;
			for (int k = 0; k < dataSet.m; k++) {
				sum += dataSet.A[i * dataSet.m + k] * dataSet.B[k * dataSet.p + j];
			}
			dataSet.C[i * dataSet.p + j] = sum;
		}
	}
}

void multiply2D(DataSet dataSet) {
	#pragma omp parallel for
	for (int ij = 0; ij < dataSet.n * dataSet.p; ij++) {
		int i = ij / dataSet.p;
		int j = ij % dataSet.p;
		int sum = 0;
		for (int k = 0; k < dataSet.m; k++) {
			sum += dataSet.A[i * dataSet.m + k] * dataSet.B[k * dataSet.p + j];
		}
		dataSet.C[i * dataSet.p + j] = sum;
	}
}

/* serial times:
{
	128: 0.018327,
	512: 0.527193,
	1024: 4.321610
}
*/

/* 1d times:
{
	1: {
		128: 0.025206,
		512: 0.799504,
		1024: 7.230956
	},
	2: {
		128: 0.013516,
		512: 0.398704,
		1024: 3.397725
	},
	4: {
		128: 0.012529,
		512: 0.342141,
		1024: 1.908663
	},
	8: {
		128: 0.004698,
		512: 0.121051,
		1024: 1.855229
	}
}
*/

/* 2d times:
{
	1: {
		128: 0.025054,
		512: 0.753504,
		1024: 6.992076
	},
	2: {
		128: 0.013373,
		512: 0.385742,
		1024: 3.403440
	},
	4: {
		128: 0.006312,
		512: 0.128940,
		1024: 1.290252
	},
	8: {
		128: 0.006731,
		512: 0.222499,
		1024: 1.855229
	}
}
*/
