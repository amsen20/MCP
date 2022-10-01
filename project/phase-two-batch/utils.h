#ifndef __UTILS_H__
#define __UTILS_H__

#include "genann.h"
#include "conf.h"

int argmax(const double *arr, int n) {
    int ret = 0;
    for(int i=1 ; i<n ; i++)
        if(arr[i] > arr[ret])
            ret = i;
    
    return ret;
}

double get_acc(genann *ann, double *input, double *output) {
    int crt=0;
    for(int i=TRAIN ; i<NUM_ROWS ; i++) {
        const double *cur = genann_run(ann, input + i * ann->inputs);
        if(argmax(output + i * ann->outputs, ann->outputs) == argmax(cur, ann->outputs))
            crt ++;
    }

    return crt * 1.0 / TEST;
}

void swap(double *a, double *b) {
    double c = *a;
    *a = *b;
    *b = c;
}


void shuffle(double *input, double *output, int num_inputs, int num_outputs) {
    for(int i=0 ; i<NUM_ROWS ; i++) {
        int j = rand() % NUM_ROWS;
        for(int it=0 ; it<num_inputs ; it++)
            swap(input + i * num_inputs + it, input + j * num_inputs + it);
        for(int it=0 ; it<num_outputs ; it++)
            swap(output + i * num_outputs + it, output + j * num_outputs + it);
    }
}

#endif