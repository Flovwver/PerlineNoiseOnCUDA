#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"

#include <stdio.h>
#include <fstream>

const unsigned int height = 1024;
const unsigned int width = 1024;

cudaError_t addWithCuda(float *c);
bool saveArrayInTxt(float* array);

__device__ float Frac(float xFloat) 
{
    int xInt = xFloat;
    return xFloat - xInt;
}

__device__ float Dot(float2 vectorLeft, float2 vectorRight)
{
    return vectorLeft.x * vectorRight.x + vectorLeft.y * vectorRight.y;
}

__device__ float Rand(float2 x)
{
    float2 randomNumber = make_float2(Frac(sin(Dot(x, make_float2(78.233, 12.9898))) * 43758.5453), Frac(sin(Dot(x, make_float2(78.233 * 2, 12.9898 * 2))) * 43758.5453));
    return abs(randomNumber.x + randomNumber.y) * 0.5;
}

__device__ float Rand(int2 xInt)
{
    float2 xFloat;
    xFloat.x = xInt.x;
    xFloat.y = xInt.y;
    Rand(xFloat);
}

__device__ float Noise(int2 uv)
{
    float randomNumber = Rand(uv);
    return randomNumber;
}

__device__ float BilinearInterpolation(float f00, float f01, float f10, float f11, int x0, int x1, int y0, int y1, float2 uv)
{
    float fR1 = (x1 - uv.x) / (x1 - x0) * f00 + (uv.x - x0) / (x1 - x0) * f10;
    float fR2 = (x1 - uv.x) / (x1 - x0) * f01 + (uv.x - x0) / (x1 - x0) * f11;
    return (y1 - uv.y) / (y1 - y0) * fR1 + (uv.y - y0) / (y1 - y0) * fR2;
}

__device__ float ShellInterpolation(float2 uv, int coeficient)
{
    return BilinearInterpolation(Noise(make_int2(uv.x * coeficient, uv.y * coeficient)), Noise(make_int2(uv.x * coeficient, uv.y * coeficient+1)),
        Noise(make_int2(uv.x * coeficient + 1, uv.y * coeficient)), Noise(make_int2(uv.x * coeficient+1, uv.y * coeficient+1)), uv.x * coeficient,
        uv.x * coeficient + 1, uv.y * coeficient, uv.y * coeficient + 1, make_float2(uv.x * coeficient, uv.y * coeficient));
}

__device__ float PerlinNoise(float2 uv)
{
    float color = 0.f;
    int numberOfCycles = 10;
    for (int i = numberOfCycles; i >= 1; i--) 
    {
        color = color * 0.4 + ShellInterpolation(uv, pow(3, i));
    }
    return color * 0.8;
}

__global__ void addKernel(float *c)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    c[i* width + j] = Frac(PerlinNoise(make_float2((float) i / height, (float) j / width)));
}

int main()
{
    float* c = new float[height * width];

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    saveArrayInTxt(c);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c)
{
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<height, width>>>(dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    
    return cudaStatus;
}

bool saveArrayInTxt(float* array) 
{
    FILE* file;
    if ((file = fopen("perlineNoise.bin", "w")) == NULL) {
        printf("error\n");
        return false;
    }
    else {
        fwrite(array, sizeof(float), height * width, file);
    }
    fclose(file);
    printf("Successful \n");
    return true;
}
