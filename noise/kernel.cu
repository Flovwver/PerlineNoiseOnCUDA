#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"

#include <stdio.h>
#include <fstream>

const unsigned int height = 1024;
const unsigned int width = 1024;

cudaError_t GeneratePerlineNoise(float *c);
bool SaveArrayInTxt(float* array);
void OutputArray(float* arrayOfElements, int firsOtputElement, int lastOtputElement);

__device__ float Frac(float xFloat) 
{
    int xInt = fabs(xFloat);
    return fabs(xFloat) - xInt;
}

__device__ float Dot(float2 vectorLeft, float2 vectorRight)
{
    return vectorLeft.x * vectorRight.x + vectorLeft.y * vectorRight.y;
}

__device__ float Rand(float2 seed)
{
    float a = 12.9898;

    float b = 78.233;

    float c = 43758.5453;

    float dt = (seed.x + 5) * a + (seed.y + 5) * b;

    float sn = dt;

    return Frac(sin(sn) * c);
}

__device__ float Rand(int2 xInt)
{
    float2 xFloat;
    xFloat.x = xInt.x;
    xFloat.y = xInt.y;
    return Rand(xFloat);
}

__device__ float GenerateNoiseWithResolution(int2 uv)
{
    float randomNumber = Rand(uv);
    return randomNumber;
}

__device__ double cubicInterpolate(double p[4], double x) {
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] -
        p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

__device__ double bicubicInterpolate(double p[4][4], double x, double y) {
    double arr[4];
    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}

__device__ float BilinearInterpolation(float f00, float f01, float f10, float f11, int x0, int x1, int y0, int y1, float2 uv)
{
    float fR1 = (x1 - uv.x) / (x1 - x0) * f00 + (uv.x - x0) / (x1 - x0) * f10;
    float fR2 = (x1 - uv.x) / (x1 - x0) * f01 + (uv.x - x0) / (x1 - x0) * f11;
    return (y1 - uv.y) / (y1 - y0) * fR1 + (uv.y - y0) / (y1 - y0) * fR2;
}

__device__ float GenerateOctaveWithBicubic(float2 uv, int coeficient)
{
    double p[4][4];

    for (int i = -1; i < 3; i++) {
        for (int j = -1; j < 3; j++) {
            int2 coordInt = make_int2(uv.x * coeficient, uv.y * coeficient);
            coordInt.x += i * coeficient;
            coordInt.y += j * coeficient;
            p[i + 1][j + 1] = GenerateNoiseWithResolution(coordInt);
        }
    }

    return bicubicInterpolate(p, uv.x, uv.y);
}

__device__ float GenerateOctaveWithBilinear(float2 uv, int coeficient)
{
    return BilinearInterpolation(GenerateNoiseWithResolution(make_int2(uv.x * coeficient, uv.y * coeficient)), GenerateNoiseWithResolution(make_int2(uv.x * coeficient, uv.y * coeficient+1)),
        GenerateNoiseWithResolution(make_int2(uv.x * coeficient + 1, uv.y * coeficient)), GenerateNoiseWithResolution(make_int2(uv.x * coeficient+1, uv.y * coeficient+1)), uv.x * coeficient,
        uv.x * coeficient + 1, uv.y * coeficient, uv.y * coeficient + 1, make_float2(uv.x * coeficient, uv.y * coeficient));
}

__device__ float GeneratePerlinNoise(float2 uv)
{
    float color = 0.f;
    int numberOfCycles = 3;
    for (int i = numberOfCycles-1; i >= numberOfCycles - 1; i--)
    {
        color = color * 0.5f + GenerateOctaveWithBicubic(uv, pow(2, i));
    }
    if (color > 1.f)
        color = 0.99f;
    return color;
}

__global__ void addKernel(float *c)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    c[i * width + j] = GeneratePerlinNoise(make_float2((float) i / height, (float) j / width));
}

int main()
{
    float* c = new float[height * width];

    cudaError_t cudaStatus = GeneratePerlineNoise(c);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Main function of Cuda failed!");
        return 1;
    }

    SaveArrayInTxt(c);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t GeneratePerlineNoise(float *c)
{
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
   
    cudaStatus = cudaMalloc((void**)&dev_c, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


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

bool SaveArrayInTxt(float* arrayOfElements) 
{
    FILE* file;
    if ((file = fopen("perlineNoise.txt", "w")) == NULL) {
        printf("error\n");
        return false;
    }
    else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++)
                fprintf(file, "%f\t", arrayOfElements[i*width+j]);
            fprintf(file, "\n");
        }
    }
    fclose(file);
    printf("Successful \n");
    return true;
}

void OutputArray(float* arrayOfElements, int firsOtputElement, int lastOtputElement) {
    for (int i = firsOtputElement; i <= lastOtputElement; i++) {
        printf("%f\t", arrayOfElements[i]);
    }
    printf("\n");
}
