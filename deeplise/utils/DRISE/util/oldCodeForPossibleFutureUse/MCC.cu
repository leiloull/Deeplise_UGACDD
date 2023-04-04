#include "MCC.cuh"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

/*
CUDA KERNELS
*/
__global__ void updateTPTNFPFN(int numInteractions, float threshold[3], ulonglong4* TPTNFPFN, float* scores, bool* interactions){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long globalID = blockId * blockDim.x + threadIdx.x;
  if(globalID < numInteractions){
    __shared__ ulonglong4 block_TPTNFPFN[3];
    for(int i = 0; i < 3; ++i) block_TPTNFPFN[i] = {0,0,0,0};
    __syncthreads();
    int eval = 0;
    eval = (eval << 1) + interactions[globalID];
    eval = (eval << 1) + (scores[globalID] > threshold[threadIdx.y]);

    switch(eval){
      case 0:{
        atomicAdd(&block_TPTNFPFN[threadIdx.y].y, 1);
      }
      case 1:{
        atomicAdd(&block_TPTNFPFN[threadIdx.y].z, 1);
      }
      case 2:{
        atomicAdd(&block_TPTNFPFN[threadIdx.y].w, 1);
      }
      case 3:{
        atomicAdd(&block_TPTNFPFN[threadIdx.y].x, 1);
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    if(block_TPTNFPFN[threadIdx.y].x){
      atomicAdd(&TPTNFPFN[threadIdx.y].x, block_TPTNFPFN[threadIdx.y].x);
    }
    if(block_TPTNFPFN[threadIdx.y].y){
      atomicAdd(&TPTNFPFN[threadIdx.y].y, block_TPTNFPFN[threadIdx.y].y);
    }
    if(block_TPTNFPFN[threadIdx.y].z){
      atomicAdd(&TPTNFPFN[threadIdx.y].z, block_TPTNFPFN[threadIdx.y].z);
    }
    if(block_TPTNFPFN[threadIdx.y].w){
      atomicAdd(&TPTNFPFN[threadIdx.y].w, block_TPTNFPFN[threadIdx.y].w);
    }
  }
}
__global__ void updateTPTNFPFN(int numInteractions, int focus, float threshold, ulonglong4* TPTNFPFN, float* scores, bool* interactions){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long globalID = blockId * blockDim.x + threadIdx.x;
  if(globalID < numInteractions){
    __shared__ ulonglong4 block_TPTNFPFN;
    block_TPTNFPFN = {0,0,0,0};
    __syncthreads();
    int eval = 0;
    eval = (eval << 1) + interactions[globalID];
    eval = (eval << 1) + (scores[globalID] > threshold);

    switch(eval){
      case 0:{
        atomicAdd(&block_TPTNFPFN.y, 1);
      }
      case 1:{
        atomicAdd(&block_TPTNFPFN.z, 1);
      }
      case 2:{
        atomicAdd(&block_TPTNFPFN.w, 1);
      }
      case 3:{
        atomicAdd(&block_TPTNFPFN.x, 1);
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    if(block_TPTNFPFN.x){
      atomicAdd(&TPTNFPFN[focus].x, block_TPTNFPFN.x);
    }
    if(block_TPTNFPFN.y){
      atomicAdd(&TPTNFPFN[focus].y, block_TPTNFPFN.y);
    }
    if(block_TPTNFPFN.z){
      atomicAdd(&TPTNFPFN[focus].z, block_TPTNFPFN.z);
    }
    if(block_TPTNFPFN.w){
      atomicAdd(&TPTNFPFN[focus].w, block_TPTNFPFN.w);
    }
  }
}

/*
CONSTRUCTORS
*/
MCC::MCC(){
  this->bspScores = NULL;
  this->interactions = NULL;
  for(int i = 0; i < 3; ++i) this->TPTNFPFN_host[i] = {0,0,0,0};
  this->TPTNFPFN_device = NULL;
  CudaSafeCall(cudaMalloc((void**)&this->TPTNFPFN_device, 3*sizeof(ulonglong4)));
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_device, this->TPTNFPFN_host, 3*sizeof(ulonglong4), cudaMemcpyHostToDevice));
  this->threshold = {0.49,0.5,0.51};
  this->ratio = {0.0,0.0,0.0};
  this->deltaThreshold = 0.01;
  this->iterations = 0;
  this->vacancyAlignment = 1;
}
MCC::~MCC(){
  if(this->TPTNFPFN_device != NULL){
    CudaSafeCall(cudaFree(this->TPTNFPFN_device));
  }
}

/*
CPU METHODS
*/
void MCC::calculateMCC(){
  if(this->TPTNFPFN_device == NULL){
    throw NullUnityException("cannot calculateMCC without positive and negatives");
  }
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_host, this->TPTNFPFN_device, 3*sizeof(ulonglong4), cudaMemcpyDeviceToHost));
  ulonglong4 lower = this->TPTNFPFN_host[0];
  ulonglong4 middle = this->TPTNFPFN_host[1];
  ulonglong4 upper = this->TPTNFPFN_host[2];
  this->ratio.x = double(((lower.x*lower.y)-(lower.z*lower.w))/sqrt((lower.x+lower.z)*(lower.x+lower.w)*(lower.y+lower.z)*(lower.y+lower.w)));
  this->ratio.y = double(((middle.x*middle.y)-(middle.z*middle.w))/sqrt((middle.x+middle.z)*(middle.x+middle.w)*(middle.y+middle.z)*(middle.y+middle.w)));
  this->ratio.z = double(((upper.x*upper.y)-(upper.z*upper.w))/sqrt((upper.x+upper.z)*(upper.x+upper.w)*(upper.y+upper.z)*(upper.y+upper.w)));
  if(std::isinf(this->ratio.x)){
    std::cout<<"WARNING: MCC lower bound calculation x/0 = inf...setting to 0"<<std::endl;
    this->ratio.x = 0.0;
  }
  if(std::isinf(this->ratio.y)){
    std::cout<<"WARNING: MCC middle bound calculation x/0 = inf...setting to 0"<<std::endl;
    this->ratio.y = 0;
  }
  if(std::isinf(this->ratio.z)){
    std::cout<<"WARNING: MCC upper bound calculation x/0 = inf...setting to 0"<<std::endl;
    this->ratio.z = 0;
  }
}
void MCC::updateCriteria(Unity* bspScores, Unity* interactions){
  if(bspScores == NULL || interactions == NULL){
    throw NullUnityException("bspScores or interactions in MCC updateCriteria");
  }
  else if(bspScores->numElements != interactions->numElements){
    throw ISEException_runtime("in MCC::updateCriteria bspScores and interactions must have same length and be aligned");
  }
  int numElements = bspScores->numElements;
  MemoryState origin[2] = {bspScores->state, interactions->state};
  bspScores->transferMemoryTo(cpu);
  float2 minMax = {FLT_MAX, 0.0f};
  float* scores_host = (float*) bspScores->host;
  for(int i = 0; i < numElements; ++i){
    if(scores_host[i] < minMax.x) minMax.x = scores_host[i];
    if(scores_host[i] > minMax.y) minMax.y = scores_host[i];
  }

  bspScores->setMemoryState(gpu);
  interactions->setMemoryState(gpu);
  float* scores_device = (float*) bspScores->device;
  bool* interactions_device = (bool*) interactions->device;
  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};
  getGrid((numElements/32) + 1, grid);

  if(this->vacancyAlignment == 1){
    float lowerMidUpperThreshold[3] = {
      this->threshold.x,
      this->threshold.y,
      this->threshold.z
    };
    for(int i = 0; i < 3; ++i){
      lowerMidUpperThreshold[i] *= (minMax.y - minMax.x);
      lowerMidUpperThreshold[i] += minMax.x;
    }
    block.y = 3;
    updateTPTNFPFN<<<grid,block>>>(numElements, lowerMidUpperThreshold, this->TPTNFPFN_device, scores_device, interactions_device);
    cudaDeviceSynchronize();
    CudaCheckError();
  }
  else{
    float currentThreshold = 0.0f;
    if(this->vacancyAlignment == 0) currentThreshold = this->threshold.x;
    else currentThreshold = this->threshold.z;
    currentThreshold *= (minMax.y - minMax.x);
    currentThreshold += minMax.x;
    updateTPTNFPFN<<<grid,block>>>(numElements, this->vacancyAlignment, currentThreshold, this->TPTNFPFN_device, scores_device, interactions_device);
    cudaDeviceSynchronize();
    CudaCheckError();
  }

  bspScores->setMemoryState(origin[0]);
  interactions->setMemoryState(origin[1]);
}
int MCC::computeIteration(){
  int directive = 0;
  this->calculateMCC();

  float upperEval = this->ratio.z - this->ratio.y;
  float lowerEval = this->ratio.x - this->ratio.y;

  if(upperEval >= 0 && lowerEval < 0){
    directive = 1;
    this->diffMCC = upperEval;
    this->moveThresholdRight();
  }
  else if(lowerEval >= 0 && upperEval < 0){
    directive = -1;
    this->diffMCC = lowerEval;
    this->moveThresholdLeft();
  }
  else if(lowerEval > 0 && upperEval > 0){//THIS IS NO MANS LAND AND NOT SURE HOW TO DEAL WITH THIS, currently will shift in direction of
    if(lowerEval > upperEval){
      directive = -1;
      this->diffMCC = lowerEval;
      this->moveThresholdLeft();
    }
    else{
      directive = 1;
      this->diffMCC = upperEval;
      this->moveThresholdRight();
    }
  }
  else if(lowerEval == 0 && upperEval == 0){
    this->diffMCC = 0;
    //what to do here?????
    directive = 0;//maybe not
    std::cout<<"MCC value is the same with threshold percentiles "
    <<this->threshold.x<<","<<this->threshold.y<<","<<this->threshold.z<<std::endl;
  }
  else{
    this->diffMCC = 0;
    directive = 0;
    std::cout<<"MCC max found at "<<this->threshold.y<<" percentile"<<std::endl;
  }
  this->iterations++;
  return directive;
}

/*
GETTERS AND SETTERS
*/
void MCC::moveThresholdRight(){
  if(this->deltaThreshold == 0){
    throw ISEException_runtime("cannot move threshold in MCC without deltaThreshold");
  }
  this->threshold.x = this->threshold.y;
  this->threshold.y = this->threshold.z;
  this->threshold.z = this->threshold.y + this->deltaThreshold;
  this->TPTNFPFN_host[0] = this->TPTNFPFN_host[1];
  this->TPTNFPFN_host[1] = this->TPTNFPFN_host[2];
  this->TPTNFPFN_host[2] = {0,0,0,0};
  if(this->TPTNFPFN_device == NULL){
    CudaSafeCall(cudaMalloc((void**)&this->TPTNFPFN_device, 3*sizeof(ulonglong4)));
  }
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_device, this->TPTNFPFN_host, 3*sizeof(ulonglong4), cudaMemcpyHostToDevice));
  this->ratio.x = this->ratio.y;
  this->ratio.y = this->ratio.z;
  this->ratio.z = 0.0;
  this->vacancyAlignment = 2;
}
void MCC::moveThresholdLeft(){
  if(this->deltaThreshold == 0){
    throw ISEException_runtime("cannot move threshold in MCC without deltaThreshold");
  }
  this->threshold.z = this->threshold.y;
  this->threshold.y = this->threshold.x;
  this->threshold.x = this->threshold.y - this->deltaThreshold;
  this->TPTNFPFN_host[2] = this->TPTNFPFN_host[1];
  this->TPTNFPFN_host[1] = this->TPTNFPFN_host[0];
  this->TPTNFPFN_host[0] = {0,0,0,0};
  if(this->TPTNFPFN_device == NULL){
    CudaSafeCall(cudaMalloc((void**)&this->TPTNFPFN_device, 3*sizeof(ulonglong4)));
  }
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_device, this->TPTNFPFN_host, 3*sizeof(ulonglong4), cudaMemcpyHostToDevice));
  this->ratio.z = this->ratio.y;
  this->ratio.y = this->ratio.x;
  this->ratio.x = 0.0;
  this->vacancyAlignment = 0;
}
void MCC::moveThresholdRight(float deltaThreshold){
  this->deltaThreshold = deltaThreshold;
  if(this->deltaThreshold == 0){
    throw ISEException_runtime("cannot move threshold in MCC without deltaThreshold");
  }
  this->threshold.x = this->threshold.y;
  this->threshold.y = this->threshold.z;
  this->threshold.z = this->threshold.y + this->deltaThreshold;
  this->TPTNFPFN_host[0] = this->TPTNFPFN_host[1];
  this->TPTNFPFN_host[1] = this->TPTNFPFN_host[2];
  this->TPTNFPFN_host[2] = {0,0,0,0};
  if(this->TPTNFPFN_device == NULL){
    CudaSafeCall(cudaMalloc((void**)&this->TPTNFPFN_device, 3*sizeof(ulonglong4)));
  }
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_device, this->TPTNFPFN_host, 3*sizeof(ulonglong4), cudaMemcpyHostToDevice));
  this->ratio.x = this->ratio.y;
  this->ratio.y = this->ratio.z;
  this->ratio.z = 0.0;
  this->vacancyAlignment = 2;
}
void MCC::moveThresholdLeft(float deltaThreshold){
  this->deltaThreshold = deltaThreshold;
  if(this->deltaThreshold == 0){
    throw ISEException_runtime("cannot move threshold in MCC without deltaThreshold");
  }
  this->threshold.z = this->threshold.y;
  this->threshold.y = this->threshold.x;
  this->threshold.x = this->threshold.y - this->deltaThreshold;
  this->TPTNFPFN_host[2] = this->TPTNFPFN_host[1];
  this->TPTNFPFN_host[1] = this->TPTNFPFN_host[0];
  this->TPTNFPFN_host[0] = {0,0,0,0};
  if(this->TPTNFPFN_device == NULL){
    CudaSafeCall(cudaMalloc((void**)&this->TPTNFPFN_device, 3*sizeof(ulonglong4)));
  }
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_device, this->TPTNFPFN_host, 3*sizeof(ulonglong4), cudaMemcpyHostToDevice));
  this->ratio.z = this->ratio.y;
  this->ratio.y = this->ratio.x;
  this->ratio.x = 0.0;
  this->vacancyAlignment = 0;
}
void MCC::setDeltaThreshold(float deltaThreshold){
  this->deltaThreshold = deltaThreshold;
}
void MCC::setThreshold(float threshold){
  this->threshold = {
    threshold-this->deltaThreshold,
    threshold,
    threshold+this->deltaThreshold
  };
  this->reset();
}
void MCC::setThreshold(float3 threshold){
  this->threshold = threshold;
  this->reset();
}
void MCC::reset(){
  this->ratio = {0.0,0.0,0.0};
  for(int i = 0; i < 3; ++i) this->TPTNFPFN_host[i] = {0,0,0,0};
  if(this->TPTNFPFN_device == NULL){
    CudaSafeCall(cudaMalloc((void**)&this->TPTNFPFN_device, 3*sizeof(ulonglong4)));
  }
  CudaSafeCall(cudaMemcpy(this->TPTNFPFN_device, this->TPTNFPFN_host, 3*sizeof(ulonglong4), cudaMemcpyHostToDevice));
  this->iterations = 0;
  this->vacancyAlignment = 1;
}
