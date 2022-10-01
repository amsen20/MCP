#include <cuda_runtime.h>
#include <stdio.h>

#include "kernels.h"
#include "cu_utils.h"

__global__ void
mat_mul_vector_kernel(
    int n,
    int m,
    double *mat,
    double *vec,
    double *out,
    int offset=1,
    double val=-1.0
) {
    __shared__ double x[BLOCK];
    
    int row = blockIdx.x;
    int col = threadIdx.x;
    x[col] = (col < m ? (*(mat + row * m + col) * (col >= offset ? vec[col-offset] : val)) : 0);


    __syncthreads();

    for(int i=BLOCK/2 ; i>0 ; i /= 2) {
        if(col < i && col + i < m)
            x[col] += x[col + i];
        __syncthreads();
    }

    if(!col) {
        out[row] = x[0];
    }
}

__global__ void sigmoid(
    double *arr
) {
    int tid = threadIdx.x;
    double x = arr[tid];
    if (x < -15.0) {
        arr[tid] = 0;
        return;
    }
    if (x > 15.0) {
        arr[tid] = 1;
        return;
    }

    arr[tid] = 1.0 / (1 + exp(-x));
}

__global__ void print_out(
    double *arr,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
}

void 
mat_mul_vector_sig(
    int n,
    int m,
    double *mat,
    double *vec,
    double *out,
    int offset,
    double val
) {
    mat_mul_vector_kernel <<< n, m >>> (n, m, mat, vec, out, offset, val);
    sigmoid <<< 1, n >>> (out);
}


void
printall(double *arr, int n) {
    print_out <<<(n + 1023)/1024, 1024>>> (arr, n);
}

__global__ void
transpose_kernel(
    int n,
    int m,
    double *mat,
    int offset,
    double *omat
) {
    __shared__ double scr[CHUNCK][CHUNCK];
    int strow = blockIdx.x * blockDim.x;

    int stcol = blockIdx.y * blockDim.y;

    int x = threadIdx.y, y = threadIdx.x;

    if(x + strow < n && y + stcol < m)
        scr[y][x] = mat[(strow + x) * (m + offset) + stcol + y + offset];
    __syncthreads();
    if(x + stcol < m && y + strow < n) {
        omat[(x + stcol) * n + (y + strow)] = scr[x][y];
    }
}

__global__ void
apply_kernel(
    double *o,
    double *arr
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    arr[tid] = o[tid] * (1 - o[tid]) * arr[tid];
}

void
calc_delta(
    int n,
    int m,
    double *mat,
    int offset,
    double *vec,
    double *out,
    double *O,
    double *scratch
) {
    dim3 grid((n + CHUNCK - 1) / CHUNCK, (m + CHUNCK - 1)/CHUNCK, 1);
    dim3 block(CHUNCK, CHUNCK, 1);
    transpose_kernel <<<grid, block>>> (n, m, mat, offset, scratch);
    mat_mul_vector_kernel <<<m, n>>> (m, n, scratch, vec, out, 0, 0);
    apply_kernel <<<1, m>>> (O, out);
}

__global__ void
upd_mat_kernel(
    int n,
    int m,
    double *mat,
    double *v,
    double *u,
    int offset
) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    if(x < n && y < m)
        mat[x * m + y] += v[x] * LEARNING_RATE * (y >= offset ? u[y-offset] : -1.0);
}

void
upd_mat(
    int n,
    int m,
    double *mat,
    double *v,
    double *u,
    int offset
) {
    upd_mat_kernel <<< n, m >>> (n, m, mat, v, u, offset);
}

__global__ void
diff_and_apply_kernel(
    double *a,
    double *b,
    double *out
) {
    int tid = threadIdx.x;
    out[tid] = (a[tid] - b[tid]) * b[tid] * (1 - b[tid]);
}

void
diff_and_apply(
    double *a,
    double *b,
    double *out,
    int n
) {
    diff_and_apply_kernel <<< 1, n >>> (a, b, out);
}
