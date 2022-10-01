#ifndef __KERNEL_H__
#define __KERNEL_H__

void 
mat_mul_vector_sig(
    int n,
    int m,
    double *mat,
    double *vec,
    double *out,
    int offset=1,
    double val=-1.0
);

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
);

void
upd_mat(
    int n,
    int m,
    double *mat,
    double *v,
    double *u,
    int offset
);

void
diff_and_apply(
    double *a,
    double *b,
    double *out,
    int n
);

void printall(
    double *arr,
    int n
);

#endif