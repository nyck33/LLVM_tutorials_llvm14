# on GPU and Compiler Engineering for LLM workloads

### Aren't LLM's computational graphs static in nature?

For large transformer-based models like BERT, GPT, Llama, and Google's Bard, the computational graph is typically static in nature. These models are characterized by their regular structure and use of matrix multiplications, making them easier to optimize. The input sequence length might vary, but the overall architecture stays the same.

### Here are some points to consider for these types of models:

### Fixed Computational Graph
The architecture is generally fixed, with well-defined layers and operations that don't change during inference or training. This makes it easier to perform optimizations ahead of time (AOT).

### Data Parallelism
Since these models are often too large to fit into a single GPU's memory, data parallelism strategies are commonly used. In this case, each GPU gets a mini-batch of the data, and the gradients are aggregated across GPUs.

### Pipeline Parallelism
For extremely large models, even a single layer might not fit into GPU memory. Pipeline parallelism can be used to split layers across multiple GPUs.

### Model Parallelism
The model parameters are distributed across multiple GPUs, and each GPU computes a part of the forward and backward pass.

### Communication Overhead
When using multiple GPUs, especially in a cluster, communication between the GPUs (or between nodes in a cluster) can become a bottleneck. Optimizations that reduce the amount of data that needs to be communicated can be beneficial.  However I am delegating this part to SnuRHAC.  

### Tensor Core Utilization
Modern GPUs have specialized hardware for accelerating certain types of mixed-precision matrix multiplications and convolutions, known as Tensor Cores. Properly optimizing the model to leverage Tensor Cores can result in significant speed-ups.

In the context of a multi-GPU setup managed by something like SnuRhac, which presents the multi-GPU cluster as a single virtual GPU, many of the challenges around data and model parallelism might be abstracted away from the user. However, the underlying system would still need to manage these aspects efficiently.

For these types of models and setups, JIT and trace-based optimizations might be less relevant than in scenarios with dynamically changing workloads. However, they could still be used for tasks like kernel fusion, which can reduce the number of operations and therefore improve performance. But generally, optimizations for these types of models would focus more on efficient parallelization strategies and communication optimizations.

## However, some points to consider below:

## Dynamic Batching during training 
Some frameworks use dynamic batching techniques to group sequences of similar lengths together, which minimizes the amount of padding needed. However, this results in batches that can have different shapes, affecting the computational graph and potentially triggering recompilation if JIT is being used.

## Compiler Engineer's Role (LLVM-based Optimization):

### 1. Generate specialized kernels for different batch sizes

In the context of machine learning, a "kernel" refers to a small unit of computation optimized for a specific operation or sequence of operations. For instance, a kernel might be responsible for performing a certain layer's computation in a neural network. 

When you're dealing with dynamic batching, where the batch size may change over time, you could have a single general-purpose kernel that works for any batch size. However, this kernel might not be optimized for any specific size, leading to suboptimal performance. 

As a compiler engineer, you can create multiple specialized kernels optimized for different batch sizes. For instance, you could have one kernel optimized for batch sizes between 1 and 16, another for batch sizes between 17 and 64, and so on. The optimization could involve loop unrolling, using different memory access patterns, or other machine-specific optimizations.

Here's a simplified example. Suppose you have a loop in LLVM IR that looks something like:

```llvm
for i = 0, i < batchSize, i++
  // do something
```

For a batch size of 8, you could unroll this loop, like so:

```llvm
for i = 0, i < 8, i += 4
  // do something for i
  // do something for i+1
  // do something for i+2
  // do something for i+3
```

### 2. Fuse operations in the computation graph to reduce memory traffic

The computation for a neural network can be represented as a directed acyclic graph where each node is an operation (like addition, multiplication, or a more complex operation like convolution), and each edge is a multi-dimensional array (or tensor) flowing between operations.

Operation fusion involves combining multiple operations into a single operation, thereby reducing the amount of data that needs to be read from and written to memory. This can be particularly important for GPUs, where memory bandwidth is often the bottleneck.

For example, suppose you have two operations in your computation graph: one that multiplies a tensor \( A \) by a scalar \( \alpha \), and another that adds a tensor \( B \) to the result. Normally, this would involve reading \( A \) from memory, multiplying it by \( \alpha \), writing the result back to memory, reading that result plus \( B \) from memory, adding them together, and writing the final result back to memory.

Through operation fusion, you could combine these into a single operation that reads \( A \) and \( B \) from memory, performs \( A \times \alpha + B \), and writes the result back to memory, all in one go.

In LLVM, you could accomplish this by recognizing the sequence of instructions corresponding to these separate operations and replacing them with a new sequence of instructions that performs the combined operation. This would be a part of the LLVM optimization passes.

By applying these kinds of optimizations, you can make the most of the hardware's capabilities and improve the efficiency of your machine learning models.

### GPU Engineer's Role:
Custom CUDA Code:
- Implement batch-aware shared memory utilization to minimize global memory access.
- Use asynchronous memory copies for overlapping computation and data transfer.

##### Example Before Optimization (simplified CUDA code for matrix multiplication):
```cuda
__global__ void matmul(float *A, float *B, float *C, int N) {
    // ... standard matrix multiplication
}
```
#### after
```cuda
__global__ void batch_aware_matmul(float *A, float *B, float *C, int N, int batchSize) {
    int dynamicTileDim = (batchSize < 100) ? 16 : 32;  // Choose tile size based on batch size
    
    extern __shared__ float shared_mem[];
    float *tileA = shared_mem;
    float *tileB = &shared_mem[dynamicTileDim * dynamicTileDim];

    // The rest of the matrix multiplication logic
    // use dynamicTileDim in place of TILE_DIM
}
```
### For the kernel launch, you can dynamically allocate the shared memory as follows:
```cuda
int dynamicTileDim = (batchSize < 100) ? 16 : 32;
size_t sharedMemSize = 2 * dynamicTileDim * dynamicTileDim * sizeof(float);
batch_aware_matmul<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, N, batchSize);
```
#### This way, the tile size (dynamicTileDim) is determined dynamically based on the batch size, and shared memory is allocated accordingly. The batch-aware shared memory utilization is implemented by choosing a smaller or larger tile size based on the batch size.

#### what are tileA and tileB?

In this example, tileA and tileB are supposed to be two separate "tiles" or sub-matrices that fit into the shared memory. These tiles are used to store portions of the larger matrices A and B for the purpose of performing the matrix multiplication in a more cache-efficient way.

The shared memory array shared_mem is allocated to be large enough to hold both tileA and tileB. The pointer tileA is essentially pointing to the start of shared_mem, and tileB is pointing to the location immediately following where tileA ends in shared_mem.

```bash
|----- tileA -----|----- tileB -----|
^                 ^                 ^
|                 |                 |
shared_mem    tileA            tileB
```

By setting tileB to start dynamicTileDim * dynamicTileDim elements after tileA, we are effectively partitioning the shared memory into two separate regions: one for tileA and another for tileB. This allows us to make efficient use of shared memory for storing these two tiles.

## Variable Workloads at Inference time
In real-world applications, you may not have control over the sequence lengths that are being sent to your model for inference. This variability can affect both performance and accuracy, and might require dynamic adjustments to the model or the data pipeline.

## Compiler Engineer's Role (LLVM-based Optimization):

### 1. Monitoring / Profiling 

Here, you'd collect runtime statistics on sequence lengths. This could be done with simple counters or more advanced telemetry.

```c++
std::unordered_map<int, int> sequence_length_counter;

void monitor_sequence_length(int length) {
    sequence_length_counter[length]++;
}
```

### 2. Triggering

Once you have enough data, you could decide whether to trigger JIT compilation for a new sequence length. You might do this based on a threshold or a time interval.

```c++
void check_for_new_optimization() {
    for (auto &[length, count]: sequence_length_counter) {
        if (count > SOME_THRESHOLD) {
            trigger_jit_compilation(length);
        }
    }
}
```

### 3. JIT Compilation

Here, you'd use LLVM's JIT capabilities to compile specialized code for the new sequence length. The exact details would depend on your specific use-case and LLVM's API.

Reference: [LLVM JIT](https://llvm.org/docs/tutorial/BuildingAJIT1.html)

```c++
void trigger_jit_compilation(int length) {
    // Use LLVM API to generate IR
    // Compile IR to machine code using JIT
    // Update the dispatch table
}
```

### 4. Update Dispatch Table

Here you could use a dispatch table or a similar mechanism to keep track of which compiled function to call for each sequence length.

```c++
std::unordered_map<int, std::function<void()>> dispatch_table;

void update_dispatch_table(int length, std::function<void()> optimized_func) {
    dispatch_table[length] = optimized_func;
}
```

### 5. Runtime Execution

Finally, at runtime, you'd use your dispatch table to call the appropriate version of the function.

```c++
void execute(int length) {
    if (dispatch_table.find(length) != dispatch_table.end()) {
        dispatch_table[length]();
    } else {
        // Fallback to generic implementation
    }
}
```

This is a very simplified example, but it should give you an idea of how you could use LLVM's JIT compilation capabilities to optimize your code at runtime based on observed workloads.

## GPU Engineer's Role
Custom CUDA Code:
- Implement dynamic kernel launches that adjust shared memory and thread block sizes based on the sequence length.
- Use warp-level primitives for better efficiency when dealing with divergent sequence lengths within a batch.

### Example Before Optimization (simplified CUDA code for transformer layer):

```cuda
__global__ void transformer_layer(float *input, float *output, int seq_len) {
    // ... standard transformer layer computation
}
```
### After Optimization (sequence-length aware), simple example that sums elements along each sequence:
```cuda
__global__ void dynamic_transformer_layer(float *input, float *output, int seq_len, int batch_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Calculate global index in the array
    int idx = bx * blockDim.x + tx;

    // Make sure we don't read/write out of bounds
    if (idx < seq_len * batch_size) {
        // Initialize shared memory
        extern __shared__ float shared_mem[];
        shared_mem[tx] = 0;

        // Assume each sequence is stored contiguously in the input array
        int sequence_start = (idx / seq_len) * seq_len;

        // Sum over the sequence length
        for (int i = 0; i < seq_len; ++i) {
            shared_mem[tx] += input[sequence_start + i];
        }

        // Write to output
        output[idx] = shared_mem[tx];
    }
}

```
### What makes up a batch at inference time for LLMs?

In the context of deep learning models like Transformers, a single "batch" could indeed be just one sequence, especially if that sequence is already close to the maximum token limit. Batching at inference time is usually different from training time. During training, you often have control over the size and makeup of each batch, but during inference, the "batch size" is often dictated by the number of incoming queries. Sometimes sequences are batched together to improve throughput, but this is generally easier when the sequences are of similar length to minimize padding.

For variable-length sequences at inference time, you would typically pad each sequence to the nearest multiple of some fixed block size (which could be as small as 1 for no batching) and then dispatch each padded sequence (or batch of sequences) to the model separately. This allows each query to be processed independently, with its own sequence length.

In such a scenario, the CUDA kernel responsible for a certain layer in the model might need to know the actual sequence length to avoid doing unnecessary work on the padded tokens. Here, the seq_len parameter would be variable from one kernel invocation to the next, and the kernel would use this parameter to determine how much work needs to be done for each query.

# Appendix


### In this example, each thread block computes a 16x16 sub-matrix of C by accumulating the product of 16x16 sub-matrices from A and B. Since each thread needs to access multiple elements from these sub-matrices, storing them in shared memory can reduce the number of global memory accesses, which are more time-consuming.


```cuda
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

```