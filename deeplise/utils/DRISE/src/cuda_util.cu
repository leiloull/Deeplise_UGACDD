#include "cuda_util.cuh"

__constant__ float pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;


__device__ __host__ float dist(const float3 &a, const float3 &b){
  return sqrtf(((a.x-b.x)*(a.x-b.x)) + ((a.y-b.y)*(a.y-b.y)) + ((a.z-b.z)*(a.z-b.z)));
}

__device__ __host__ bool operator<(const float3 &a, const float3 &b){
  return (a.x < b.x) && (a.y < b.y) && (a.z < b.z);
}
__device__ __host__ bool operator>(const float3 &a, const float3 &b){
  return (a.x > b.x) && (a.y > b.y) && (a.z > b.z);
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__device__ __host__ float3 operator/(const float3 &a, const float3 &b) {
  return {a.x/b.x, a.y/b.y, a.z/b.z};
}
__device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z};
}
__device__ __host__ float dotProduct(const float3 &a, const float3 &b){
  return (a.x*b.x) + (a.y*b.y) + (a.z*b.z);
}
__device__ __host__ float3 operator+(const float3 &a, const float &b){
  return {a.x+b, a.y+b, a.z+b};
}
__device__ __host__ float3 operator-(const float3 &a, const float &b){
  return {a.x-b, a.y-b, a.z-b};
}
__device__ __host__ float3 operator/(const float3 &a, const float &b){
  return {a.x/b, a.y/b, a.z/b};
}
__device__ __host__ float3 operator*(const float3 &a, const float &b){
  return {a.x*b, a.y*b, a.z*b};
}
__device__ __host__ float3 operator+(const float &a, const float3 &b) {
  return {a+b.x, a+b.y, a+b.z};
}
__device__ __host__ float3 operator-(const float &a, const float3 &b) {
  return {a-b.x, a-b.y, a-b.z};
}
__device__ __host__ float3 operator/(const float &a, const float3 &b) {
  return {a/b.x, a/b.y, a/b.z};
}
__device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return {a*b.x, a*b.y, a*b.z};
}
__device__ __host__ bool operator==(const float3 &a, const float3 &b){
  return (a.x==b.x)&&(a.y==b.y)&&(a.z==b.z);
}

__device__ __host__ float2 operator+(const float2 &a, const float2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ float2 operator-(const float2 &a, const float2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ float2 operator/(const float2 &a, const float2 &b){
  return {a.x / b.x, a.y / b.y};
}
__device__ __host__ float2 operator*(const float2 &a, const float2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ float dotProduct(const float2 &a, const float2 &b){
  return (a.x*b.x) + (a.y*b.y);
}
__device__ __host__ float2 operator+(const float2 &a, const float &b){
  return {a.x + b, a.y + b};
}
__device__ __host__ float2 operator-(const float2 &a, const float &b){
  return {a.x - b, a.y - b};
}
__device__ __host__ float2 operator/(const float2 &a, const float &b){
  return {a.x / b, a.y / b};
}
__device__ __host__ float2 operator*(const float2 &a, const float &b){
  return {a.x * b, a.y * b};
}
__device__ __host__ float2 operator+(const float &a, const float2 &b){
  return {a + b.x, a + b.y};
}
__device__ __host__ float2 operator-(const float &a, const float2 &b){
  return {a - b.x, a - b.y};
}
__device__ __host__ float2 operator/(const float &a, const float2 &b){
  return {a / b.x, a / b.y};
}
__device__ __host__ float2 operator*(const float &a, const float2 &b){
  return {a * b.x, a * b.y};
}
__device__ __host__ bool operator==(const float2 &a, const float2 &b){
  return a.x == b.x && a.y == b.y;
}

__device__ __host__ float2 operator+(const float2 &a, const int2 &b){
  return {a.x + (float) b.x, a.y + (float) b.y};
}
__device__ __host__ float2 operator-(const float2 &a, const int2 &b){
  return {a.x - (float) b.x, a.y - (float) b.y};
}
__device__ __host__ float2 operator/(const float2 &a, const int2 &b){
  return {a.x / (float) b.x, a.y / (float) b.y};
}
__device__ __host__ float2 operator*(const float2 &a, const int2 &b){
  return {a.x * (float) b.x, a.y * (float) b.y};
}
__device__ __host__ float2 operator+(const int2 &a, const float2 &b){
  return {(float) a.x + b.x, (float) a.y + b.y};
}
__device__ __host__ float2 operator-(const int2 &a, const float2 &b){
  return {(float) a.x - b.x, (float) a.y - b.y};
}
__device__ __host__ float2 operator/(const int2 &a, const float2 &b){
  return {(float) a.x / b.x, (float) a.y / b.y};
}
__device__ __host__ float2 operator*(const int2 &a, const float2 &b){
  return {(float) a.x * b.x, (float) a.y * b.y};
}

__device__ __host__ int2 operator+(const int2 &a, const int2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ int2 operator-(const int2 &a, const int2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ float2 operator/(const int2 &a, const int2 &b){
  return {(float) a.x / (float) b.x, (float) a.y / (float) b.y};
}
__device__ __host__ int2 operator*(const int2 &a, const int2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ int dotProduct(const int2 &a, const int2 &b){
  return (a.x*b.x) + (a.y*b.y);
}
__device__ __host__ int2 operator+(const int2 &a, const int &b){
  return {a.x + b, a.y + b};
}
__device__ __host__ int2 operator-(const int2 &a, const int &b){
  return {a.x - b, a.y - b};
}
__device__ __host__ float2 operator/(const int2 &a, const int &b){
  return {(float) a.x / (float) b, (float) a.y / (float) b};
}
__device__ __host__ int2 operator*(const int2 &a, const int &b){
  return {a.x * b, a.y * b};
}
__device__ __host__ int dotProduct(const int2 &a, const int &b){
  return (a.x * b) + (a.y * b);
}

__device__ __host__ float3 operator+(const int3 &a, const float3 &b){
  return {((float)a.x) + b.x, ((float)a.y) + b.y, ((float)a.z) + b.z};
}
__device__ __host__ float3 operator-(const int3 &a, const float3 &b){
  return {((float)a.x) - b.x, ((float)a.y) - b.y, ((float)a.z) - b.z};
}
__device__ __host__ float3 operator/(const int3 &a, const float3 &b){
  return {((float)a.x) / b.x, ((float)a.y) / b.y, ((float)a.z) / b.z};
}
__device__ __host__ float3 operator*(const int3 &a, const float3 &b){
  return {((float)a.x) * b.x, ((float)a.y) * b.y, ((float)a.z) * b.z};
}
__device__ __host__ float3 operator+(const float3 &a, const int3 &b){
  return {a.x + ((float)b.x), a.y + ((float)b.y), a.z + ((float)b.z)};
}
__device__ __host__ float3 operator-(const float3 &a, const int3 &b){
  return {a.x - ((float)b.x), a.y - ((float)b.y), a.z - ((float)b.z)};
}
__device__ __host__ float3 operator/(const float3 &a, const int3 &b){
  return {a.x / ((float)b.x), a.y / ((float)b.y), a.z / ((float)b.z)};
}
__device__ __host__ float3 operator*(const float3 &a, const int3 &b){
  return {a.x * ((float)b.x), a.y * ((float)b.y), a.z * ((float)b.z)};
}
__device__ __host__ float3 operator+(const int3 &a, const float &b){
  return {((float)a.x) + b, ((float)a.y) + b, ((float)a.z) + b};
}
__device__ __host__ float3 operator-(const int3 &a, const float &b){
  return {((float)a.x) - b, ((float)a.y) - b, ((float)a.z) - b};
}
__device__ __host__ float3 operator/(const int3 &a, const float &b){
  return {((float)a.x) / b, ((float)a.y) / b, ((float)a.z) / b};
}
__device__ __host__ float3 operator*(const int3 &a, const float &b){
  return {((float)a.x) * b, ((float)a.y) * b, ((float)a.z) * b};
}
__device__ __host__ float3 operator+(const float &a, const int3 &b){
  return {a + ((float)b.x), a + ((float)b.y), a + ((float)b.z)};
}
__device__ __host__ float3 operator-(const float &a, const int3 &b){
  return {a - ((float)b.x), a - ((float)b.y), a - ((float)b.z)};
}
__device__ __host__ float3 operator/(const float &a, const int3 &b){
  return {a / ((float)b.x), a / ((float)b.y), a / ((float)b.z)};
}
__device__ __host__ float3 operator*(const float &a, const int3 &b){
  return {a * ((float)b.x), a * ((float)b.y), a * ((float)b.z)};
}


void max_occupancy(dim3 &grid, dim3 &block, const int &gridDim, const int &blockDim, const uint3 &forceBlock, const long &valueToAchieve){

}

void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block) {
  if(2147483647l > size){
    grid.x = size;
    return;
  }
  else if(2147483647l * 1024l > size){
    grid.x = 2147483647l;
    block.x = 1024l;
    while(block.x * grid.x > size){
      block.x--;
    }
    block.x++;
  }
  else{
    grid.x = 65535l;
    block.x = 1024l;
    grid.y = 1;
    while(grid.x * grid.y * block.x < size){
      grid.y++;
    }
  }
}
void getGrid(unsigned long size, dim3 &grid) {
  if(2147483647l > size){
    grid.x = size;
  }
  else{
    grid.x = 65535l;
    grid.y = 1;
    while(grid.x * grid.y * grid.y < size){
      grid.y++;
    }
  }
}



void printDeviceProperties() {
  std::cout<<"\n---------------START OF DEVICE PROPERTIES---------------\n"<<std::endl;

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf(" -Device name: %s\n\n", prop.name);
    printf(" -Memory\n  -Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  -Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
    printf("  -Peak Memory Bandwidth (GB/s): %f\n",2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  -Total Global Memory (bytes): %lo\n", prop.totalGlobalMem);
    printf("  -Total Const Memory (bytes): %lo\n", prop.totalConstMem);
    printf("  -Max pitch allowed for memcpy in regions allocated by cudaMallocPitch() (bytes): %lo\n\n", prop.memPitch);
    printf("  -Shared Memory per block (bytes): %lo\n", prop.sharedMemPerBlock);
    printf("  -Max number of threads per block: %d\n",prop.maxThreadsPerBlock);
    printf("  -Max number of blocks: %dx%dx%d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  -32bit Registers per block: %d\n", prop.regsPerBlock);
    printf("  -Threads per warp: %d\n\n", prop.warpSize);
    printf("  -Total number of Multiprocessors: %d\n",prop.multiProcessorCount);
    printf("  -Shared Memory per Multiprocessor (bytes): %lo\n",prop.sharedMemPerMultiprocessor);
    printf("  -32bit Registers per Multiprocessor: %d\n\n", prop.regsPerMultiprocessor);
    printf("  -Number of asynchronous engines: %d\n", prop.asyncEngineCount);
    printf("  -Texture alignment requirement (bytes): %lo\n  -Texture base addresses that are aligned to "
    "textureAlignment bytes do not need an offset applied to texture fetches.\n\n", prop.textureAlignment);

    printf(" -Device Compute Capability:\n  -Major revision #: %d\n  -Minor revision #: %d\n", prop.major, prop.minor);
    printf(" -Run time limit for kernels that get executed on this device: ");
    if(prop.kernelExecTimeoutEnabled){
      printf("YES\n");
    }
    else{
      printf("NO\n");
    }
    printf(" -Device is ");
    if(prop.integrated){
      printf("integrated. (motherboard)\n");
    }
    else{
      printf("discrete. (card)\n\n");
    }
    if(prop.isMultiGpuBoard){
      printf(" -Device is on a MultiGPU configurations.\n\n");
    }
    switch(prop.computeMode){
      case(0):
        printf(" -Default compute mode (Multiple threads can use cudaSetDevice() with this device)\n");
        break;
      case(1):
        printf(" -Compute-exclusive-thread mode (Only one thread in one processwill be able to use\n cudaSetDevice() with this device)\n");
        break;
      case(2):
        printf(" -Compute-prohibited mode (No threads can use cudaSetDevice() with this device)\n");
        break;
      case(3):
        printf(" -Compute-exclusive-process mode (Many threads in one process will be able to use\n cudaSetDevice() with this device)\n");
        break;
      default:
        printf(" -GPU in unknown compute mode.\n");
        break;
      }
      if(prop.canMapHostMemory){
        printf("\n -The device can map host memory into the CUDA address space for use with\n cudaHostAlloc() or cudaHostGetDevicePointer().\n\n");
      }
      else{
        printf("\n -The device CANNOT map host memory into the CUDA address space.\n\n");
      }
      printf(" -ECC support: ");
      if(prop.ECCEnabled){
        printf(" ON\n");
      }
      else{
        printf(" OFF\n");
      }
      printf(" -PCI Bus ID: %d\n", prop.pciBusID);
      printf(" -PCI Domain ID: %d\n", prop.pciDomainID);
      printf(" -PCI Device (slot) ID: %d\n", prop.pciDeviceID);
      printf(" -Using a TCC Driver: ");
      if(prop.tccDriver){
        printf("YES\n");
      }
      else{
        printf("NO\n");
      }
    }
    std::cout<<"\n----------------END OF DEVICE PROPERTIES----------------\n"<<std::endl;
}
