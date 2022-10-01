nvcc main.cu genann.cu kernels.cu -arch=sm_75 -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcuda -lm -Xcompiler -fopenmp -O3 && echo compiled && ./a.out 784 1 512 10 ../MNIST.data

