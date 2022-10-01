#ifndef __UTILS_H__
#define __UTILS_H__

#include "genann.h"
#include "conf.h"
#include "cu_utils.h"

int argmax(const double *arr, int n) {
    int ret = 0;
    for(int i=1 ; i<n ; i++)
        if(arr[i] > arr[ret])
            ret = i;
    
    return ret;
}

double get_acc(genann *ann, double *input, double *output) {
    int crt=0;
    double cur[ann->outputs];

    for(int i=TRAIN ; i<NUM_ROWS ; i++) {

        double *d_cur = genann_run_parallel(ann, input + i * ann->inputs);
        ensure(cudaMemcpy(cur, d_cur, ann->outputs * sizeof(double), cudaMemcpyDeviceToHost), "couldn't copy.");
        
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

#endif