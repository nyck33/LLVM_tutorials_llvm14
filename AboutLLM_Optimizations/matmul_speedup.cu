__global__ void matMul(float* A, float* B, float* C, int N) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.y * 16 + ty;
    int j = blockIdx.x * 16 + tx;

    float sum = 0;

    for (int k = 0; k < N; k += 16) {
        sA[ty][tx] = A[i * N + k + tx];
        sB[ty][tx] = B[(k + ty) * N + j];
        __syncthreads();

        //directive hints to compiler to unroll the loop to reduce the loop control overhead
        #pragma unroll
        
        for (int m = 0; m < 16; ++m) {
            //results of 16 multiplications and 15 additiionns involving 32 data fetches from shared memory rather than global memory which is much faster
            sum += sA[ty][m] * sB[m][tx];
        }
        __syncthreads();
    }

    C[i * N + j] = sum;
}

/*
In this example, each thread block computes a 16x16 sub-matrix of C by accumulating the product of 16x16 sub-matrices from A and B. Since each thread needs to access multiple elements from these sub-matrices, storing them in shared memory can reduce the number of global memory accesses, which are more time-consuming.
*/