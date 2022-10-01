#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>

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

    int _;
    char *path = argv[5];
    FILE *in = fopen(path, "r");
    for(int i=0 ; i<NUM_ROWS ; i++) {
        for(int j=0 ; j<num_inputs ; j++)
            fscanf(in, "%lf", input + i * num_inputs + j);
        for(int j=0 ; j<num_outputs ; j++)
            fscanf(in, "%lf", output + i * num_outputs + j);
    }
    fclose(in);

    double *d_input, *d_output, *scratch;
    // copy input to device
    {
        int input_size = sizeof(double)*NUM_ROWS*num_inputs;
        ensure(cudaMalloc((void**)&d_input, input_size), "could not allocate in device.");
        ensure(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice), "could not copy to device");
    }
    // copy output to device
    {
        int output_size = sizeof(double)*NUM_ROWS*num_outputs;
        ensure(cudaMalloc((void**)&d_output, output_size), "could not allocate in device.");
        ensure(cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice), "could not copy to device");
    }

    // allocate scratch
    {
        ensure(cudaMalloc((void**)&scratch, SCRATCH_SIZE), "could not allocate in device.");
    }

    genann *ann = genann_init(num_inputs, num_hidden_layers, num_hidden_layers_neurons, num_outputs);

    for(int epoch=0 ; epoch<EPOCH_NUM ; epoch ++) {
        for(int i=0 ; i<TRAIN ; i++)
            genann_train_parallel(ann, d_input + i*num_inputs, d_output + i*num_outputs, scratch); 
    }

    printf("acc: %lf\n", get_acc(ann, d_input, output));
    
    genann_free(ann);
    ensure(cudaFree(d_input), "could not free");
    ensure(cudaFree(d_output), "could not free");
    ensure(cudaFree(scratch), "could not free");

    {
        double elapsedtime = omp_get_wtime() - starttime;
	    printf("Time Elapsed: %f\n", elapsedtime);
    }
}