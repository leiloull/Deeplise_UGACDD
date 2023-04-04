#ifndef MCC_CUH
#define MCC_CUH
#include "common_includes.h"
#include "cuda_util.cuh"
#include "Unity.cuh"

/*
CUDA kernels
*/
__global__ void updateTPTNFPFN(int numInteractions, float threshold[3], ulonglong4* TPTNFPFN, float* scores, bool* interactions);
__global__ void updateTPTNFPFN(int numInteractions, int focus, float threshold, ulonglong4* TPTNFPFN, float* scores, bool* interactions);

class MCC{

ulonglong4 TPTNFPFN_host[3];
ulonglong4* TPTNFPFN_device;
float3 threshold;//percentile...default = {0.49,0.5,0.51}
double3 ratio;
float deltaThreshold;//percentile...default = 0.01
int vacancyAlignment;//if 0 then lower needs computing, if 1 all need computing, if 2 upper needs computing
float diffMCC;

Unity* bspScores;
Unity* interactions;
void calculateMCC();


public:

unsigned int iterations;

MCC();
MCC(float initialThresholdGuess);
MCC(float initialThresholdGuess, float deltaThreshold);
~MCC();

void updateCriteria(Unity* bspScores, Unity* interactions);
int computeIteration();

void moveThresholdRight();
void moveThresholdLeft();
void moveThresholdRight(float deltaThreshold);
void moveThresholdLeft(float deltaThreshold);
void setDeltaThreshold(float deltaThreshold);
void setThreshold(float threshold);
void setThreshold(float3 threshold);
void reset();

};


#endif /* MCC_CUH */
