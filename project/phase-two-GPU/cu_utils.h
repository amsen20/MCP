#ifndef __CU_UTILS_H__
#define __CU_UTILS_H__

#ifndef GENANN_RANDOM
/* We use the following for uniform random numbers between 0 and 1.
 * If you have a better function, redefine this macro. */
#define GENANN_RANDOM() (((double)rand())/RAND_MAX)
#endif

#include "conf.h"

#include <cuda_runtime.h>

inline void ensure(cudaError_t error, const char *msg) {
    if(error != cudaSuccess) {
        fprintf(stderr, "%s ", msg);
        fprintf(stderr, " ---- error code: %s\n", cudaGetErrorString(error));
        
        exit(EXIT_FAILURE);
    }
}

#endif
