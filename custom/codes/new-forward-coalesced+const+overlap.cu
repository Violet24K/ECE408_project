#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__constant__ float Mask[6000];

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // int numblock_eachcolumn = (H_out - 1)/TILE_WIDTH + 1;
    int numblock_eachrow = (W_out - 1)/TILE_WIDTH + 1;
    int w_out = TILE_WIDTH * (bz % numblock_eachrow) + tx;
    int h_out = TILE_WIDTH * (bz/numblock_eachrow) + ty;
    int b_out = bx;
    int m_out = by;

    if (h_out < H_out && w_out < W_out)
    {
        float result = 0;

        for (int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    result += x4d(b_out, c, h_out + p, w_out + q) * k4d(m_out, c, p, q);
                }
            }
        }
        y4d(b_out, m_out, h_out, w_out) = result;
    }
    





#undef y4d
#undef x4d
#undef k4d
#undef sm2d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
    int SegSize = 10;
    
    cudaMalloc((void **) device_y_ptr, (B * M * (H - K + 1) * (W - K + 1))*sizeof(float));
    cudaMalloc((void **) device_x_ptr, (B * C * H * W)*sizeof(float));

    // Use multiple stream
    cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8, stream9;
    cudaStreamCreate( &stream0);
    cudaStreamCreate( &stream1);
    cudaStreamCreate( &stream2);
    cudaStreamCreate( &stream3);
    cudaStreamCreate( &stream4);
    cudaStreamCreate( &stream5);
    cudaStreamCreate( &stream6);
    cudaStreamCreate( &stream7);
    cudaStreamCreate( &stream8);
    cudaStreamCreate( &stream9);

    
    int SegSize_x = C * H * W;
    //int SegSize_k = C * K * K;
    int SegSize_y = M * (H - K + 1) * (W - K + 1);
    //cudaMemcpy(*device_x_ptr, host_x, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask, host_k, (M * C * K * K)*sizeof(float));
    
    dim3 dimGrid(SegSize, M, ceil((float)(H - K + 1)/TILE_WIDTH)*ceil((float)(W - K + 1)/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    for (int i = 0; i < B; i +=  10 * SegSize)
    {
        cudaMemcpyAsync(*device_x_ptr + (i + 0 * SegSize) * SegSize_x, host_x + (i + 0 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(*device_x_ptr + (i + 1 * SegSize) * SegSize_x, host_x + (i + 1 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(*device_x_ptr + (i + 2 * SegSize) * SegSize_x, host_x + (i + 2 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream2);
        cudaMemcpyAsync(*device_x_ptr + (i + 3 * SegSize) * SegSize_x, host_x + (i + 3 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream3);
        cudaMemcpyAsync(*device_x_ptr + (i + 4 * SegSize) * SegSize_x, host_x + (i + 4 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream4);
        cudaMemcpyAsync(*device_x_ptr + (i + 5 * SegSize) * SegSize_x, host_x + (i + 5 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream5);
        cudaMemcpyAsync(*device_x_ptr + (i + 6 * SegSize) * SegSize_x, host_x + (i + 6 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream6);
        cudaMemcpyAsync(*device_x_ptr + (i + 7 * SegSize) * SegSize_x, host_x + (i + 7 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream7);
        cudaMemcpyAsync(*device_x_ptr + (i + 8 * SegSize) * SegSize_x, host_x + (i + 8 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream8);
        cudaMemcpyAsync(*device_x_ptr + (i + 9 * SegSize) * SegSize_x, host_x + (i + 9 * SegSize) * SegSize_x, SegSize * SegSize_x * sizeof(float), cudaMemcpyHostToDevice,stream9);

        //cudaMemcpyAsync(*device_k_ptr, host_k + (i + 0) * SegSize_k, SegSize_k * sizeof(float), .., stream0);
        //cudaMemcpyAsync(*device_k_ptr, host_k + (i + 1) * SegSize_k, SegSize_k * sizeof(float), .., stream1);
        //cudaMemcpyAsync(*device_k_ptr, host_k + (i + 2) * SegSize_k, SegSize_k * sizeof(float), .., stream2);
        //cudaMemcpyAsync(*device_k_ptr, host_k + (i + 3) * SegSize_k, SegSize_k * sizeof(float), .., stream3);
        //cudaMemcpyAsync(*device_k_ptr, host_k + (i + 4) * SegSize_k, SegSize_k * sizeof(float), .., stream4);


        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream0>>>(*device_y_ptr + (i + 0 * SegSize) * SegSize_y, *device_x_ptr + (i + 0 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream1>>>(*device_y_ptr + (i + 1 * SegSize) * SegSize_y, *device_x_ptr + (i + 1 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream2>>>(*device_y_ptr + (i + 2 * SegSize) * SegSize_y, *device_x_ptr + (i + 2 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream3>>>(*device_y_ptr + (i + 3 * SegSize) * SegSize_y, *device_x_ptr + (i + 3 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream4>>>(*device_y_ptr + (i + 4 * SegSize) * SegSize_y, *device_x_ptr + (i + 4 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream5>>>(*device_y_ptr + (i + 5 * SegSize) * SegSize_y, *device_x_ptr + (i + 5 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream6>>>(*device_y_ptr + (i + 6 * SegSize) * SegSize_y, *device_x_ptr + (i + 6 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream7>>>(*device_y_ptr + (i + 7 * SegSize) * SegSize_y, *device_x_ptr + (i + 7 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream8>>>(*device_y_ptr + (i + 8 * SegSize) * SegSize_y, *device_x_ptr + (i + 8 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream9>>>(*device_y_ptr + (i + 9 * SegSize) * SegSize_y, *device_x_ptr + (i + 9 * SegSize) * SegSize_x, *device_k_ptr, B, M, C, H, W, K);


        cudaMemcpyAsync(host_y + (i + 0 * SegSize) * SegSize_y, *device_y_ptr + (i + 0 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(host_y + (i + 1 * SegSize) * SegSize_y, *device_y_ptr + (i + 1 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(host_y + (i + 2 * SegSize) * SegSize_y, *device_y_ptr + (i + 2 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(host_y + (i + 3 * SegSize) * SegSize_y, *device_y_ptr + (i + 3 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream3);
        cudaMemcpyAsync(host_y + (i + 4 * SegSize) * SegSize_y, *device_y_ptr + (i + 4 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream4);
        cudaMemcpyAsync(host_y + (i + 5 * SegSize) * SegSize_y, *device_y_ptr + (i + 5 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream5);
        cudaMemcpyAsync(host_y + (i + 6 * SegSize) * SegSize_y, *device_y_ptr + (i + 6 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream6);
        cudaMemcpyAsync(host_y + (i + 7 * SegSize) * SegSize_y, *device_y_ptr + (i + 7 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream7);
        cudaMemcpyAsync(host_y + (i + 8 * SegSize) * SegSize_y, *device_y_ptr + (i + 8 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream8);
        cudaMemcpyAsync(host_y + (i + 9 * SegSize) * SegSize_y, *device_y_ptr + (i + 9 * SegSize) * SegSize_y, SegSize * SegSize_y * sizeof(float), cudaMemcpyDeviceToHost, stream9);






    }




    


}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    //dim3 dimGrid(B, M, ceil((float)(H - K + 1)/TILE_WIDTH)*ceil((float)(W - K + 1)/TILE_WIDTH));
    //dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    //conv_forward_kernel<<<dimGrid, dimBlock, TILE_WIDTH * TILE_WIDTH * C * sizeof(float)>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    //cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
