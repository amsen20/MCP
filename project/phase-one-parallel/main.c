#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "conf.h"
#include "utils.h"
#include "genann.h"

int main(int argc, char *argv[]) {
    double starttime = omp_get_wtime();
    int num_inputs = atoi(argv[1]);
    int num_hidden_layers = atoi(argv[2]);
    int num_hidden_layers_neurons = atoi(argv[3]);
    int num_outputs = atoi(argv[4]);
    double *input = (double*)malloc(NUM_ROWS * num_inputs * sizeof(double));
    double *output = (double*)malloc(NUM_ROWS * num_outputs * sizeof(double));
    
    char *path = argv[5];
    FILE *in = fopen(path, "r");
    for(int i=0 ; i<NUM_ROWS ; i++) {
        for(int j=0 ; j<num_inputs ; j++)
            fscanf(in, "%lf", input + i * num_inputs + j);
        for(int j=0 ; j<num_outputs ; j++)
            fscanf(in, "%lf", output + i * num_outputs + j);
    }
    fclose(in);

    genann *ann = genann_init(num_inputs, num_hidden_layers, num_hidden_layers_neurons, num_outputs);

    for(int epoch=0 ; epoch<EPOCH_NUM ; epoch ++) {
        for(int i=0 ; i<TRAIN ; i++)
            genann_train_parallel(ann, input + i * num_inputs, output + i * num_outputs, LEARNING_RATE);
    }
    
    printf("acc: %lf\n", get_acc(ann, input, output));
    genann_free(ann);

    double elapsedtime = omp_get_wtime() - starttime;
	printf("Time Elapsed: %f\n", elapsedtime);
}