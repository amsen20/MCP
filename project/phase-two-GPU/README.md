# GPU-accelerated neural network

In this project, [genann](https://github.com/codeplea/genann) is accelerated by running the backpropagation algorithm on the GPU.\
For running the algorithm on GPU, memory coalescing, vectorize operations, reduction, and other techniques are used.\
You can find kernels in [kernels.cu](kernels.cu) file.\
A 2x speedup is gained on an MX450 Nvidia GPU.


Checkout the [presentation file](../MP%20project.pdf) for more information.
