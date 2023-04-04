#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH

#include "common_includes.h"

extern __constant__ float pi;

//use generics
void printDeviceProperties();
__device__ __host__ bool operator<(const float3 &a, const float3 &b);
__device__ __host__ bool operator>(const float3 &a, const float3 &b);

__device__ __host__ float dist(const float3 &a, const float3 &b);

__device__ __host__ float3 operator+(const float3 &a, const float3 &b);
__device__ __host__ float3 operator-(const float3 &a, const float3 &b);
__device__ __host__ float3 operator/(const float3 &a, const float3 &b);
__device__ __host__ float3 operator*(const float3 &a, const float3 &b);
__device__ __host__ float dotProduct(const float3 &a, const float3 &b);
__device__ __host__ float3 operator+(const float3 &a, const float &b);
__device__ __host__ float3 operator-(const float3 &a, const float &b);
__device__ __host__ float3 operator/(const float3 &a, const float &b);
__device__ __host__ float3 operator*(const float3 &a, const float &b);
__device__ __host__ float3 operator+(const float &a, const float3 &b);
__device__ __host__ float3 operator-(const float &a, const float3 &b);
__device__ __host__ float3 operator/(const float &a, const float3 &b);
__device__ __host__ float3 operator*(const float &a, const float3 &b);

__device__ __host__ float2 operator+(const float2 &a, const float2 &b);
__device__ __host__ float2 operator-(const float2 &a, const float2 &b);
__device__ __host__ float2 operator/(const float2 &a, const float2 &b);
__device__ __host__ float2 operator*(const float2 &a, const float2 &b);
__device__ __host__ float dotProduct(const float2 &a, const float2 &b);
__device__ __host__ float2 operator+(const float2 &a, const float &b);
__device__ __host__ float2 operator-(const float2 &a, const float &b);
__device__ __host__ float2 operator/(const float2 &a, const float &b);
__device__ __host__ float2 operator*(const float2 &a, const float &b);
__device__ __host__ float2 operator+(const float &a, const float2 &b);
__device__ __host__ float2 operator-(const float &a, const float2 &b);
__device__ __host__ float2 operator/(const float &a, const float2 &b);
__device__ __host__ float2 operator*(const float &a, const float2 &b);
__device__ __host__ bool operator==(const float2 &a, const float2 &b);

__device__ __host__ float2 operator+(const float2 &a, const int2 &b);
__device__ __host__ float2 operator-(const float2 &a, const int2 &b);
__device__ __host__ float2 operator/(const float2 &a, const int2 &b);
__device__ __host__ float2 operator*(const float2 &a, const int2 &b);
__device__ __host__ float2 operator+(const int2 &a, const float2 &b);
__device__ __host__ float2 operator-(const int2 &a, const float2 &b);
__device__ __host__ float2 operator/(const int2 &a, const float2 &b);
__device__ __host__ float2 operator*(const int2 &a, const float2 &b);

__device__ __host__ float3 operator+(const int3 &a, const float3 &b);
__device__ __host__ float3 operator-(const int3 &a, const float3 &b);
__device__ __host__ float3 operator/(const int3 &a, const float3 &b);
__device__ __host__ float3 operator*(const int3 &a, const float3 &b);
__device__ __host__ float3 operator+(const float3 &a, const int3 &b);
__device__ __host__ float3 operator-(const float3 &a, const int3 &b);
__device__ __host__ float3 operator/(const float3 &a, const int3 &b);
__device__ __host__ float3 operator*(const float3 &a, const int3 &b);
__device__ __host__ float3 operator+(const int3 &a, const float &b);
__device__ __host__ float3 operator-(const int3 &a, const float &b);
__device__ __host__ float3 operator/(const int3 &a, const float &b);
__device__ __host__ float3 operator*(const int3 &a, const float &b);
__device__ __host__ float3 operator+(const float &a, const int3 &b);
__device__ __host__ float3 operator-(const float &a, const int3 &b);
__device__ __host__ float3 operator/(const float &a, const int3 &b);
__device__ __host__ float3 operator*(const float &a, const int3 &b);

__device__ __host__ int2 operator+(const int2 &a, const int2 &b);
__device__ __host__ int2 operator+(const int2 &a, const int &b);
__device__ __host__ int2 operator-(const int2 &a, const int2 &b);
__device__ __host__ int2 operator-(const int2 &a, const int &b);
__device__ __host__ float2 operator/(const float2 &a, const int2 &b);
__device__ __host__ int2 operator*(const int2 &a, const int2 &b);
__device__ __host__ int dotProduct(const int2 &a, const int2 &b);

/*
must be in the compilation unit

__device__ __forceinline__ unsigned long getGlobalIdx_1D_1D(){
  return blockIdx.x *blockDim.x + threadIdx.x;
}
__device__ __forceinline__ unsigned long getGlobalIdx_1D_2D(){
  return blockIdx.x * blockDim.x * blockDim.y +
    threadIdx.y * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ unsigned long getGlobalIdx_1D_3D(){
  return blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
    threadIdx.z * blockDim.y * blockDim.x +
    threadIdx.y * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_2D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_3D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
    (threadIdx.z * (blockDim.x * blockDim.y)) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_3D_1D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x +
    gridDim.x * gridDim.y * blockIdx.z;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_3D_2D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x +
    gridDim.x * gridDim.y * blockIdx.z;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_3D_3D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x +
    gridDim.x * gridDim.y * blockIdx.z;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
    (threadIdx.z * (blockDim.x * blockDim.y)) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
*/

//TODO make grid and block setting max occupancy and add more overloaded methods for various situations
void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block);
void getGrid(unsigned long size, dim3 &grid);
//NOTE: not implemented
void max_occupancy(dim3 &grid, dim3 &block, const int &gridDim, const int &blockDim, const uint3 &forceBlock, const long &valueToAchieve);

void printDeviceProperties();


#endif /* CUDA_UTIL_CUH */
