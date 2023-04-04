#include "ParticleGraph.cuh"

__device__ __forceinline__ float atomicMinFloat (float * addr, float value){
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value){
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __host__ __forceinline__ float square(const float &a){
  return a*a;
}
__device__ __host__ __forceinline__ float elucid(const float3 &a, const float3 &b){
  return sqrtf(square(a.x - b.x) + square(a.y - b.y) + square(a.z - b.z));
}
__device__ int getEdgeIndexFromOrderedList(const int2 &edge, const int &numElements){
  int index = 0;
  index = ((numElements - 1)*numElements/2) - ((numElements - edge.x)*(numElements - edge.x - 1)/2);
  return index + (edge.y - edge.x - 1);
}
__device__ int getTriangleIndexFromOrderedList(const int3 &triangle, const int &numElements){
  int index = 0;
  int n = 0;
  for(int i = 0; i < triangle.x; ++i){
    n = numElements - 2 - triangle.x;
    index += (n*(n+1)/2);
  }
  n = numElements - triangle.x - 1;
  index += (((n-1)*n/2) - ((n - (triangle.y - triangle.x) + 1)*(n - (triangle.y - triangle.x))/2)) + (triangle.z - triangle.y - 1);
  return index;
}
__device__ int getTriangleTypeIndexFromOrderedList(const int3 &triangle, const int &numTypes){
  int index = 0;
  for(int i = 0; i < numTypes; ++i){
    if(triangle.x != i){
      for(int j = i; j < numTypes; ++j){
        index += (numTypes - j);
      }
      continue;
    }
    else{
      for(int j = i; j < numTypes; ++j){
        if(triangle.y != j){
          index += (numTypes - j);
          continue;
        }
        else {
          for(int k = j; k < numTypes; ++k){
            if(triangle.z != k){
              ++index;
            }
            else{
              return index;
            }
          }
        }
      }
    }
  }
  return -1;
}
__device__ int2 getEdgeFromIndexInOrderedList(const int &index, const int &numElements){
  int2 edge = {0,0};
  int i;
  for(i = numElements - 1;i < index; i += (numElements - edge.x - 1)){
    ++edge.x;
  }
  edge.y = numElements - (i - index);
  return edge;
}
__device__ __forceinline__ void orderInt3(int3 &toOrder){
  if(toOrder.x > toOrder.y){
    toOrder.x ^= toOrder.y;
    toOrder.y ^= toOrder.x;
    toOrder.x ^= toOrder.y;
  }
  if(toOrder.x > toOrder.z){
    toOrder.x ^= toOrder.z;
    toOrder.z ^= toOrder.x;
    toOrder.x ^= toOrder.z;
  }
  if(toOrder.y > toOrder.z){
    toOrder.y ^= toOrder.z;
    toOrder.z ^= toOrder.y;
    toOrder.y ^= toOrder.z;
  }
}

/*
TRIANGLE TRAINING KERNELS
*/

__global__ void edgeNodeFinder(const int numEdges, const int2* edges, int* edgeNodes){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(globalID < numEdges - 1){
    if(edges[globalID].x != edges[globalID+1].x){
      edgeNodes[edges[globalID+1].x] = globalID + 1;
    }
  }
}

__global__ void countTriangles(const int numSpheres, const int numEdges, const int2* edges, const int* triangleNodes, int* numTriangles){
  unsigned long globalID = getGlobalIdx_2D_1D();
  __shared__ int localSum;
  localSum = 0;
  __syncthreads();
  if(globalID < numEdges){

    int u = edges[globalID].x, v = edges[globalID].y;
    int u_it = triangleNodes[u], u_end = triangleNodes[u + 1];
    int v_it = triangleNodes[v], v_end = triangleNodes[v + 1];

    if(u_it == -1 || v_it == -1) return;
    int temp = u + 1;
    while(u_end == -1){
      if(temp + 1 >= numSpheres) return;
      u_end = triangleNodes[++temp];
    }
    temp = v + 1;
    while(v_end == -1){
      if(temp + 1 >= numSpheres) return;
      v_end = triangleNodes[++temp];
    }

    // if((u_it == 0 && u != 0) ||
    // (v_it == 0 && v != 0) ||
    // v_end == 0 || u_end == 0) return;

    int a = edges[u_it].y, b = edges[v_it].y;
    while (u_it < u_end && v_it < v_end) {
      int d = a - b;
      if (d <= 0) {
        a = edges[++u_it].y;
      }
      if (d >= 0) {
        b = edges[++v_it].y;
      }
      if (d == 0) {
        atomicAdd(&localSum, 1);
      }
    }
  }
  __syncthreads();
  atomicAdd(numTriangles, localSum);
}
__global__ void recordTriangles(const int numSpheres, const int numEdges, const int2* edges, const int* triangleNodes, int3* triangles, int* index){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(globalID < numEdges){
    int u = edges[globalID].x, v = edges[globalID].y;
    int u_it = triangleNodes[u], u_end = triangleNodes[u + 1];
    int v_it = triangleNodes[v], v_end = triangleNodes[v + 1];
    int3 newTriangle = {-1,-1,-1};

    if(u_it == -1 || v_it == -1) return;
    int temp = u + 1;
    while(u_end == -1){
      if(temp + 1 >= numSpheres) return;
      u_end = triangleNodes[++temp];
    }
    temp = v + 1;
    while(v_end == -1){
      if(temp + 1 >= numSpheres) return;
      v_end = triangleNodes[++temp];
    }

    // if((u_it == 0 && u != 0) ||
    // (v_it == 0 && v != 0) ||
    // v_end == 0 || u_end == 0) return;

    int a = edges[u_it].y, b = edges[v_it].y;
    while (u_it < u_end && v_it < v_end) {
      int d = a - b;
      if (d <= 0) {
        a = edges[++u_it].y;
      }
      if (d >= 0) {
        b = edges[++v_it].y;
      }
      if (d == 0) {
        newTriangle = {edges[globalID].x, edges[globalID].y, edges[u_it - 1].y};
        orderInt3(newTriangle);
        triangles[atomicAdd(index, 1)] = newTriangle;
      }
    }
  }
}

__global__ void triangleRefFinder(const int numTrianglePointers, const int3* __restrict__ trianglePointers, const Sphere* __restrict__ spheres, const int numSphereTypes, int* __restrict__ triangleReferences){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(globalID < numTrianglePointers){
    int3 trianglePointer = trianglePointers[globalID];
    int3 type = {(int)spheres[trianglePointer.x].type, (int)spheres[trianglePointer.y].type, (int)spheres[trianglePointer.z].type};
    if(type.x == -1 || type.y == -1 || type.z == -1){
      printf("ERROR sphere has no type %f,%f,%f\n",spheres[trianglePointer.x].center.x,spheres[trianglePointer.x].center.y,spheres[trianglePointer.x].center.z);
      asm("trap;");
    }
    orderInt3(type);
    triangleReferences[globalID] = getTriangleTypeIndexFromOrderedList(type,numSphereTypes);
  }
}

__global__ void countSurfaceOccurances(const int numTrianglePointers, const int* __restrict__ triangleReferences, int2* __restrict__ scores){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(globalID < numTrianglePointers){
    atomicAdd(&scores[triangleReferences[globalID]].x, 1);
  }
}

__global__ void countInteractionOccurances(int2* __restrict__ scores, const int minInteractions, const float threshold, const int numTrianglePointers, const int3* __restrict__ trianglePointers,
int numEdges, const int2* __restrict__ edges, const Sphere* __restrict__ targetSpheres, const Sphere* __restrict__ querySpheres, const int* __restrict__ triangleReferences){
  unsigned long blockId = blockIdx.y*gridDim.x + blockIdx.x;
  if(blockId < numTrianglePointers){
    int3 trianglePointer = trianglePointers[blockId];
    __shared__ bool foundInteraction;
    foundInteraction = false;
    __syncthreads();
    Sphere targets[3] = {targetSpheres[trianglePointer.x],targetSpheres[trianglePointer.y],targetSpheres[trianglePointer.z]};

    int2 edgePointer = {-1,-1};
    for(int edgeIndex = threadIdx.x; edgeIndex < numEdges && !foundInteraction; edgeIndex+=blockDim.x){
      edgePointer = edges[edgeIndex];
      Sphere querys[2] = {querySpheres[edgePointer.x], querySpheres[edgePointer.y]};
      int interactionCount = 0;
      float regThreshold = threshold;
      for(int t = 0; t < 3; ++t){
        for(int q = 0; q < 2; ++q){
          if(elucid(querys[q].center, targets[t].center) <= regThreshold){
            ++interactionCount;
          }
        }
      }
      if(interactionCount >= minInteractions){
        foundInteraction = true;
      }
    }
    __syncthreads();
    if(threadIdx.x == 0 && foundInteraction){
      atomicAdd(&scores[triangleReferences[blockId]].y,1);
    }
  }
}


__global__ void normalizer(const int numTriangles, Triangle* triangles, uint2 sums){
  unsigned int globalID = getGlobalIdx_2D_1D();
  if(globalID < numTriangles){
    Triangle triangle = triangles[globalID];
    if(triangle.occurances == 0) return;
    uint2 reg_sums = sums;
    // triangle.affinity = ((float)triangle.interactions/reg_sums.y)/((float)triangle.occurances/reg_sums.x);
    triangle.affinity = ((float)triangle.interactions) / ((float)triangle.occurances);
    triangles[globalID] = triangle;
  }
}
/*
BSP kernels
*/
__global__ void computeAtomScores(const int numTrianglePointers, const int* triangleReferences,
  const int3* trianglePointers, const Triangle* triangles, float* scores, int* triangleCounter){
  unsigned long globalID = getGlobalIdx_2D_1D();
  int3 triangle = {-1,-1,-1};
  if(globalID < numTrianglePointers){

    triangle = trianglePointers[globalID];
    Triangle scoredTriangle = triangles[triangleReferences[globalID]];

    if(scoredTriangle.affinity > 0.0f){
      // atomicMaxFloat(&scores[triangle.x], scoredTriangle.affinity);
      // atomicMaxFloat(&scores[triangle.y], scoredTriangle.affinity);
      // atomicMaxFloat(&scores[triangle.z], scoredTriangle.affinity);

      atomicAdd(&scores[triangle.x], scoredTriangle.affinity);
      atomicAdd(&scores[triangle.y], scoredTriangle.affinity);
      atomicAdd(&scores[triangle.z], scoredTriangle.affinity);

      atomicAdd(&triangleCounter[triangle.x], 1);
      atomicAdd(&triangleCounter[triangle.y], 1);
      atomicAdd(&triangleCounter[triangle.z], 1);
    }
  }
}


__global__ void interactionQuantification(unsigned int numTargetSpheres, Sphere* targetSpheres, unsigned int numQuerySpheres, Sphere* querySpheres, float* interactions){

  unsigned int blockId = blockIdx.x * gridDim.y + blockIdx.y;
  if(blockId < numTargetSpheres){
    __shared__ float interaction;
    interactions[blockId] = FLT_MAX;
    __syncthreads();
    Sphere sphere = targetSpheres[blockId];
    for(int p = threadIdx.x; p < numQuerySpheres; p += blockDim.x){
      atomicMinFloat(&interaction, elucid(sphere.center, querySpheres[p].center));
    }
    __syncthreads();
    if(threadIdx.x == 0) interactions[blockId] = interaction;
  }
}

__global__ void interactionQuantification(unsigned int numTargetSpheres, Sphere* targetSpheres, unsigned int numQuerySpheres, Sphere* querySpheres, bool* interactions){

  unsigned int blockId = blockIdx.x * gridDim.y + blockIdx.y;
  if(blockId < numTargetSpheres){
    __shared__ bool interaction;
    interaction = false;
    __syncthreads();
    Sphere sphere = targetSpheres[blockId];
    for(int p = threadIdx.x; p < numQuerySpheres && !interaction; p += blockDim.x){
      if(elucid(sphere.center, querySpheres[p].center) <= 6.0f){
        interaction = true;
      }
    }
    __syncthreads();
    if(threadIdx.x == 0) interactions[blockId] = interaction;
  }
}

__global__ void computeAdjacency(unsigned int numSpheres, unsigned int maxEdges, int* rows, int* columns, Sphere* spheres){
  unsigned int blockId = blockIdx.x * gridDim.y + blockIdx.y;
  if(blockId < numSpheres){
    Sphere sphere = spheres[blockId];
    Sphere sphereToCheck;
    for(int s = threadIdx.x; s < numSpheres; s+=blockDim.x){
      if(s == blockId) continue;
      sphereToCheck = spheres[s];
      if(elucid(sphere.center, sphereToCheck.center) < 2.0f){
        columns[blockId*maxEdges + atomicAdd(&rows[blockId], 1)] = s;
      }
    }
  }
}

void getSpecificPointers(Octree* octree, std::set<MoleculeType> types, int &numSpecific, unsigned int* &specificPointers){
  std::vector<unsigned int> specificVector;
  for(int i = 0; i < octree->spheres->numElements; ++i){
    for(auto type = types.begin(); type != types.end(); ++type){
      if((*type) == octree->spheres->host[i].mol_type){
        specificVector.push_back(i);
        break;
      }
    }
  }
  numSpecific = specificVector.size();
  specificPointers = new unsigned int[numSpecific];
  unsigned int* specificArray = &specificVector[0];
  std::memcpy(specificPointers, specificArray, numSpecific*sizeof(unsigned int));
}



/*
CONSTRUCTORS AND DESCTRUCTORS
*/
ParticleGraph::ParticleGraph(){
  //TODO do something here
  this->numSphereTypes = (int) atomTypeMap.size();
  this->targetSpheres = NULL;
  this->querySpheres = NULL;
  this->triangles = NULL;
}
ParticleGraph::ParticleGraph(float edgeConstraint){
  this->targetEdgeConstraints = {2.0f, edgeConstraint};
  this->queryEdgeConstraints = {2.0f, edgeConstraint};
  this->interactionThreshold = 6.0f;
  this->minInteractions = 4;
  this->numSphereTypes = (int) atomTypeMap.size();
  this->targetSpheres = NULL;
  this->querySpheres = NULL;
  this->triangles = NULL;
}
ParticleGraph::ParticleGraph(float2 edgeConstraints){
  this->targetEdgeConstraints = edgeConstraints;
  this->queryEdgeConstraints = edgeConstraints;
  this->interactionThreshold = 6.0f;
  this->minInteractions = 4;
  this->numSphereTypes = (int) atomTypeMap.size();
  this->targetSpheres = NULL;
  this->querySpheres = NULL;
  this->triangles = NULL;
}
ParticleGraph::ParticleGraph(float2 edgeConstraints, float interactionThreshold, int minInteractions){
  this->targetEdgeConstraints = edgeConstraints;
  this->queryEdgeConstraints = edgeConstraints;
  this->interactionThreshold = interactionThreshold;
  this->minInteractions = minInteractions;
  this->numSphereTypes = (int) atomTypeMap.size();
  this->targetSpheres = NULL;
  this->querySpheres = NULL;
  this->triangles = NULL;
}
ParticleGraph::ParticleGraph(float2 targetEdgeConstraints, float2 queryEdgeConstraints, float interactionThreshold, int minInteractions){
  this->targetEdgeConstraints = targetEdgeConstraints;
  this->queryEdgeConstraints = queryEdgeConstraints;
  this->interactionThreshold = interactionThreshold;
  this->minInteractions = minInteractions;
  this->numSphereTypes = (int) atomTypeMap.size();
  this->targetSpheres = NULL;
  this->querySpheres = NULL;
  this->triangles = NULL;
}
ParticleGraph::~ParticleGraph(){
  if(this->triangles != NULL) delete this->triangles;
  if(this->targetSpheres != NULL) delete this->targetSpheres;
  if(this->querySpheres != NULL) delete this->querySpheres;
}

/*
TRAINING METHODS
*/
void ParticleGraph::generateTargetTriangleReferences(int numTrianglePointers, int3* trianglePointers_device, int* &triangleReferences_device){
  CudaSafeCall(cudaMalloc((void**)&triangleReferences_device, numTrianglePointers*sizeof(int)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numTrianglePointers, grid, block);
  triangleRefFinder<<<grid, block>>>(numTrianglePointers, trianglePointers_device, this->targetSpheres->device, this->numSphereTypes, triangleReferences_device);
  CudaCheckError();

}
void ParticleGraph::generateTargetTriangles(int numEdges, int2* edges_device, int3* &trianglePointers_device, int &numTriangles){
  int* counter_device;
  numTriangles = 0;
  CudaSafeCall(cudaMalloc((void**)&counter_device, sizeof(int)));
  CudaSafeCall(cudaMemcpy(counter_device, &numTriangles, sizeof(int), cudaMemcpyHostToDevice));
  int* edgeNodes_device;
  CudaSafeCall(cudaMalloc((void**)&edgeNodes_device,this->targetSpheres->numElements*sizeof(int)));
  int* tempNodes = new int[this->targetSpheres->numElements]();
  for(int i = 0; i < this->targetSpheres->numElements; ++i){
    tempNodes[i] = -1;
  }
  CudaSafeCall(cudaMemcpy(edgeNodes_device,tempNodes, this->targetSpheres->numElements*sizeof(int),cudaMemcpyHostToDevice));
  delete[] tempNodes;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(numEdges <= 0){
    std::cout<<"ERROR there are no edges for triangle formation"<<std::endl;
    exit(-1);
  }
  getFlatGridBlock(numEdges - 1, grid, block);

  edgeNodeFinder<<<grid,block>>>(numEdges, edges_device, edgeNodes_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(numEdges, grid, block);
  countTriangles<<<grid, block>>>(this->targetSpheres->numElements, numEdges, edges_device, edgeNodes_device, counter_device);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(&numTriangles, counter_device, sizeof(int), cudaMemcpyDeviceToHost));

  int temp = 0;
  CudaSafeCall(cudaMemcpy(counter_device, &temp, sizeof(int), cudaMemcpyHostToDevice));
  int3* trianglePointers = new int3[numTriangles];
  CudaSafeCall(cudaMalloc((void**)&trianglePointers_device, numTriangles*sizeof(int3)));
  recordTriangles<<<grid,block>>>(this->targetSpheres->numElements, numEdges, edges_device, edgeNodes_device, trianglePointers_device, counter_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  std::cout<<"triangles recorded"<<std::endl;

  CudaSafeCall(cudaFree(counter_device));
  CudaSafeCall(cudaFree(edgeNodes_device));
}

void ParticleGraph::findTargetEdges(int2* &edges, int& numEdges){
  if(this->targetSpheres->numElements == 0){
    std::cout<<"ERROR cannot make target edges without target spheres set"<<std::endl;
    exit(-1);
  }
  if(this->targetSpheres->state == gpu) this->targetSpheres->transferMemoryTo(cpu);
  std::vector<int2> realEdges;
  float dist = 0.0f;
  for(int i = 0; i < this->targetSpheres->numElements; ++i){
    for(int j = i + 1; j < this->targetSpheres->numElements; ++j){
      dist = elucid(this->targetSpheres->host[i].center,this->targetSpheres->host[j].center);
      if(dist <= this->targetEdgeConstraints.y && dist >= this->targetEdgeConstraints.x){
        realEdges.push_back({i,j});
      }
    }
  }
  edges = new int2[(int) realEdges.size()];
  for(auto e = realEdges.begin(); e != realEdges.end(); ++e){
    edges[numEdges++] = (*e);
  }
}

void ParticleGraph::findQueryEdges(int2* &edges, int& numEdges){
  if(this->querySpheres->numElements == 0){
    std::cout<<"ERROR cannot make query edges without target spheres set"<<std::endl;
    exit(-1);
  }
  if(this->querySpheres->state == gpu) this->querySpheres->transferMemoryTo(cpu);
  std::vector<int2> realEdges;
  float dist = 0.0f;
  for(int i = 0; i < this->querySpheres->numElements; ++i){
    for(int j = i + 1 ; j < this->querySpheres->numElements; ++j){
      dist = elucid(this->querySpheres->host[i].center,this->querySpheres->host[j].center);
      if(dist <= this->queryEdgeConstraints.y && dist >= this->queryEdgeConstraints.x){
        realEdges.push_back({i,j});
      }
    }
  }
  edges = new int2[(int) realEdges.size()];
  for(auto e = realEdges.begin(); e != realEdges.end(); ++e){
    edges[numEdges++] = *e;
  }
}

//NOTE in the count interactions and count surface occurances, the scores variable could be replaced with the actual triangles
void ParticleGraph::executeTriangleCounters(int numTrianglePointers, int* triangleReferences_device, int3* trianglePointers_device, int numQueryEdges, int2* queryEdges_device){
  printf("graphing %d triangles with %d pairs\n",numTrianglePointers,numQueryEdges);
  time_t start = time(nullptr);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numTrianglePointers, grid, block);
  int2* scores_device;
  int2* scores = new int2[this->triangles->numElements]();

  CudaSafeCall(cudaMalloc((void**)&scores_device, this->triangles->numElements*sizeof(int2)));
  CudaSafeCall(cudaMemcpy(scores_device, scores, this->triangles->numElements*sizeof(int2), cudaMemcpyHostToDevice));

  countSurfaceOccurances<<<grid,block>>>(numTrianglePointers, triangleReferences_device, scores_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  start = time(nullptr);

  grid = {1,1,1};
  if(2147483647l > numTrianglePointers){
    grid.x = numTrianglePointers;
  }
  else{
    grid.x = 65535l;
    grid.y = 1;
    while(grid.x * grid.y  < numTrianglePointers){
      grid.y++;
    }
  }
  //grid.z = ceil((numQueryEdges + 1024)/1024);
  block = {1024,1,1};
  countInteractionOccurances<<<grid,block>>>(scores_device, this->minInteractions, this->interactionThreshold, numTrianglePointers, trianglePointers_device, numQueryEdges, queryEdges_device,
    this->targetSpheres->device, this->querySpheres->device, triangleReferences_device);
  CudaCheckError();

  CudaSafeCall(cudaMemcpy(scores, scores_device, this->triangles->numElements*sizeof(int2), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(scores_device));
  for(int i = 0; i < this->triangles->numElements; ++i){
    this->triangles->host[i].occurances += scores[i].x;
    if(scores[i].y != 0) this->triangles->host[i].interactions += scores[i].y;
  }
  delete[] scores;
  this->triangles->transferMemoryTo(gpu);
}

void ParticleGraph::updateTrianglesFromSpheres(){
  int numAtomTypes = (int) atomTypeMap.size();
  if(this->numSphereTypes == numAtomTypes && this->triangles != NULL) return;
  this->numSphereTypes = numAtomTypes;
  int possibleTriangles = 0;
  for(int i = 0; i < this->numSphereTypes; ++i){
    for(int j = i; j < this->numSphereTypes; ++j){
      for(int k = j; k < this->numSphereTypes; ++k){
        ++possibleTriangles;
      }
    }
  }

  //NOTE THIS SHOULD NOT BE NECESSARY
  if(this->triangles != NULL && this->triangles->numElements == possibleTriangles) return;

  std::vector<Triangle> metaTriangleVector;

  if(this->triangles != NULL && this->triangles->numElements != 0){
    for(int i = 0; i < this->triangles->numElements; ++i){
      metaTriangleVector.push_back(this->triangles->host[i]);
    }
    this->triangles->clear();
  }

  Triangle* triangles_host = new Triangle[possibleTriangles];
  int currentTri = 0;
  int3 currentTypes;
  for(int i = 0; i < numAtomTypes; ++i){
    for(int j = i; j < numAtomTypes; ++j){
      for(int k = j; k < numAtomTypes; ++k){
        triangles_host[currentTri] = Triangle({i,j,k});
        for(int t = 0; t < metaTriangleVector.size(); ++t){
          currentTypes = metaTriangleVector[t].atomTypes;
          if(currentTypes.x > i || currentTypes.y > j || currentTypes.z > k) continue;
          else if(i == currentTypes.x && j == currentTypes.y && k == currentTypes.z){
            if(metaTriangleVector[t].occurances != 0){
              triangles_host[currentTri].occurances =  metaTriangleVector[t].occurances;
            }
            if(metaTriangleVector[t].interactions != 0){
              triangles_host[currentTri].interactions =  metaTriangleVector[t].interactions;
            }
            if(metaTriangleVector[t].affinity != 0.0){
              triangles_host[currentTri].affinity =  metaTriangleVector[t].affinity;
            }
            break;
          }
        }
        currentTri++;
      }
    }
  }
  this->triangles = new Unity<Triangle>(triangles_host,possibleTriangles,cpu);
  this->triangles->transferMemoryTo(gpu);
}
void ParticleGraph::buildParticleGraph(ParticleList* complex){
  std::cout<<"--------------------Graph--------------------------" <<std::endl;
  time_t start = time(nullptr);

  int2* targetEdges_device = NULL;
  int2* queryEdges_device = NULL;
  int numTargetEdges = 0;
  int numQueryEdges = 0;
  int2* targetEdges = NULL;
  int2* queryEdges = NULL;

  int3* trianglePointers_device = NULL;
  int numTrianglePointers = 0;
  if(this->targetSpheres != NULL) delete this->targetSpheres;
  if(this->querySpheres != NULL) delete this->querySpheres;
  this->targetSpheres = NULL;
  this->querySpheres = NULL;

  try{
    this->targetSpheres = complex->getSpheres(false, true, this->targetType);
    this->targetSpheres->transferMemoryTo(gpu);
    this->querySpheres = complex->getSpheres(false, false, this->queryType);
    this->querySpheres->transferMemoryTo(gpu);
  }
  catch(const LackOfTypeException  &e){
    if(this->targetSpheres != NULL){
      delete this->targetSpheres;
      this->targetSpheres = NULL;
      this->querySpheres = NULL;
      throw LackOfQueryException("lack of query type when training");
    }
    else{
      this->targetSpheres = NULL;
      this->querySpheres = NULL;
      throw LackOfTargetException("lack of target type when training");
    }
  }

  std::cout << "Finding edges" << std::endl;
  this->findTargetEdges(targetEdges, numTargetEdges);
  CudaSafeCall(cudaMalloc((void**)&targetEdges_device, numTargetEdges*sizeof(int2)));
  CudaSafeCall(cudaMemcpy(targetEdges_device, targetEdges, numTargetEdges*sizeof(int2), cudaMemcpyHostToDevice));

  std::cout << "Generating Target Triangles" << std::endl;
  this->generateTargetTriangles(numTargetEdges, targetEdges_device, trianglePointers_device, numTrianglePointers);
  if(targetEdges != NULL) delete[] targetEdges;
  if(targetEdges_device != NULL) CudaSafeCall(cudaFree(targetEdges_device));
  std::cout << "Updating Triangles From Spheres" << std::endl;
  this->updateTrianglesFromSpheres();

  std::cout << "Generating Target Triangle Reference" << std::endl;
  int* triangleReferences_device;
  this->generateTargetTriangleReferences(numTrianglePointers, trianglePointers_device, triangleReferences_device);
  std::cout << "Find Query Edges" << std::endl;
  this->findQueryEdges(queryEdges, numQueryEdges);
  CudaSafeCall(cudaMalloc((void**)&queryEdges_device, numQueryEdges*sizeof(int2)));
  CudaSafeCall(cudaMemcpy(queryEdges_device, queryEdges, numQueryEdges*sizeof(int2), cudaMemcpyHostToDevice));

  std::cout << "Execute Triangel Counters" << std::endl;
  this->executeTriangleCounters(numTrianglePointers, triangleReferences_device, trianglePointers_device, numQueryEdges, queryEdges_device);

  CudaSafeCall(cudaFree(triangleReferences_device));
  if(queryEdges != NULL) delete[] queryEdges;
  if(trianglePointers_device != NULL) CudaSafeCall(cudaFree(trianglePointers_device));
  if(queryEdges_device != NULL) CudaSafeCall(cudaFree(queryEdges_device));
  std::cout << "time elapsed = " << difftime(time(nullptr), start) <<" seconds."<<std::endl;
}
void ParticleGraph::normalizeTriangles(){
  std::cout<<"-----------------Normalizing-----------------------" <<std::endl;
  if(this->triangles->numElements == 0){
    std::cout<<"ERROR no metatriangles"<<std::endl;
    exit(-1);
  }
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(this->triangles->numElements, grid, block);
  uint2 sums = {0,0};
  for(int i = 0; i < this->triangles->numElements; ++i){
    sums.x += this->triangles->host[i].occurances;
    sums.y += this->triangles->host[i].interactions;
  }
  this->triangles->transferMemoryTo(gpu);//ensure that gpu is updated
  normalizer<<<grid,block>>>(this->triangles->numElements, this->triangles->device, sums);
  CudaCheckError();
  this->triangles->transferMemoryTo(cpu);//update cpu with updated gpu triangles
}
void ParticleGraph::normalizeTriangles(float(*manipulator)(float)){
  std::cout<<"-----------------Normalizing-----------------------" <<std::endl;
  if(this->triangles->numElements == 0){
    std::cout<<"ERROR no metatriangles"<<std::endl;
    exit(-1);
  }
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(this->triangles->numElements, grid, block);
  uint2 sums = {0,0};
  for(int i = 0; i < this->triangles->numElements; ++i){
    sums.x += this->triangles->host[i].occurances;
    sums.y += this->triangles->host[i].interactions;
  }
  this->triangles->transferMemoryTo(gpu);//ensure gpu is updated
  std::cout<<sums.x<<" "<<sums.y<<std::endl;
  normalizer<<<grid,block>>>(this->triangles->numElements, this->triangles->device, sums);
  CudaCheckError();
  this->triangles->transferMemoryTo(cpu);//ensure cpu is updated with gpu triangles
  for(int i = 0; i < this->triangles->numElements; ++i){
    this->triangles->host[i].affinity = manipulator(this->triangles->host[i].affinity + 0.00001f);
  }
  this->triangles->transferMemoryTo(gpu);//ensure that this update goes to gpu
}

/*
BSP METHODS
*/
void ParticleGraph::updateScores(ParticleList* complex){
  std::cout<<"-----------------------BSP-------------------------" <<std::endl;
  if(this->triangles->numElements == 0){
    std::cout<<"ERROR cannot perform binding site prediction without trained triangles"<<std::endl;
    exit(-1);
  }
  if(this->targetSpheres != NULL) delete this->targetSpheres;

  this->targetSpheres = complex->getSpheres(false, true, this->targetType);
  this->targetSpheres->transferMemoryTo(gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int2* targetEdges_device = NULL;
  int2* targetEdges = NULL;
  int numTargetEdges = 0;

  this->findTargetEdges(targetEdges, numTargetEdges);
  CudaSafeCall(cudaMalloc((void**)&targetEdges_device, numTargetEdges*sizeof(int2)));
  CudaSafeCall(cudaMemcpy(targetEdges_device, targetEdges, numTargetEdges*sizeof(int2), cudaMemcpyHostToDevice));

  int3* trianglePointers_device = NULL;
  int numTrianglePointers = 0;
  this->generateTargetTriangles(numTargetEdges, targetEdges_device, trianglePointers_device, numTrianglePointers);
  if(targetEdges != NULL) delete[] targetEdges;
  if(targetEdges_device != NULL) CudaSafeCall(cudaFree(targetEdges_device));

  //NOTE problem here could be due to triangles being updated from spheres in buildParticleGraph but not here

  int* triangleReferences_device;
  this->generateTargetTriangleReferences(numTrianglePointers, trianglePointers_device, triangleReferences_device);

  std::cout<<"computing atom score"<<std::endl;

  float* scoreRecorder_device;
  float* scoreRecorder = new float[this->targetSpheres->numElements]();
  CudaSafeCall(cudaMalloc((void**)&scoreRecorder_device, this->targetSpheres->numElements*sizeof(float)));
  CudaSafeCall(cudaMemcpy(scoreRecorder_device, scoreRecorder, this->targetSpheres->numElements*sizeof(float), cudaMemcpyHostToDevice));

  int* triangleCounter_device;
  int* triangleCounter = new int[this->targetSpheres->numElements]();
  CudaSafeCall(cudaMalloc((void**)&triangleCounter_device, this->targetSpheres->numElements*sizeof(int)));
  CudaSafeCall(cudaMemcpy(triangleCounter_device, triangleCounter, this->targetSpheres->numElements*sizeof(int), cudaMemcpyHostToDevice));

  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(numTrianglePointers, grid, block);

  computeAtomScores<<<grid,block>>>(numTrianglePointers, triangleReferences_device, trianglePointers_device,
    this->triangles->device, scoreRecorder_device, triangleCounter_device);
  CudaCheckError();

  CudaSafeCall(cudaMemcpy(scoreRecorder, scoreRecorder_device, this->targetSpheres->numElements*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(triangleCounter, triangleCounter_device, this->targetSpheres->numElements*sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(scoreRecorder_device));
  CudaSafeCall(cudaFree(triangleCounter_device));
  CudaSafeCall(cudaFree(triangleReferences_device));
  CudaSafeCall(cudaFree(trianglePointers_device));

  int3 atomLoc = {-1,-1,-1};
  this->targetSpheres->transferMemoryTo(cpu);

  for(int s = 0; s < this->targetSpheres->numElements; ++s){
    atomLoc = this->targetSpheres->host[s].molResAtom;
    if(atomLoc.x == -1){
      std::cout<<"ERROR spheres are not calibrated"<<std::endl;
      exit(-1);
    }
    if(scoreRecorder[s] == 1.0f && !complex->molecules[atomLoc.x]->residues[atomLoc.y]->atoms[atomLoc.z]->truePositive){
      std::cout<<s<<" ERROR this should be truePositive "<<(int)this->targetSpheres->host[s].type<<" "
      <<complex->molecules[atomLoc.x]->residues[atomLoc.y]->atoms[atomLoc.z]->type<<std::endl;
      exit(-1);
    }
    
    if (triangleCounter[s] != 0) {

      complex->molecules[atomLoc.x]->residues[atomLoc.y]->atoms[atomLoc.z]->relativeAffinity = (scoreRecorder[s]/triangleCounter[s]);

    }
    else {

      complex->molecules[atomLoc.x]->residues[atomLoc.y]->atoms[atomLoc.z]->relativeAffinity = 0.0f;

    }

  }
  delete[] scoreRecorder;
  delete[] triangleCounter;
}

bool* ParticleGraph::checkInteractions(){

  bool* interactions = new bool[this->targetSpheres->numElements]();

  bool* interactions_device = NULL;


  if(this->targetSpheres->state == cpu) this->targetSpheres->transferMemoryTo(gpu);
  if(this->querySpheres->state == cpu) this->querySpheres->transferMemoryTo(gpu);

  CudaSafeCall(cudaMalloc((void**)&interactions_device, this->targetSpheres->numElements*sizeof(bool)));

  dim3 grid = {1,1,1};
  getGrid(this->targetSpheres->numElements,grid);
  dim3 block = {1024, 1, 1};
  interactionQuantification<<<grid,block>>>(this->targetSpheres->numElements,this->targetSpheres->device, this->querySpheres->numElements, this->querySpheres->device, interactions_device);
  CudaCheckError();

  CudaSafeCall(cudaMemcpy(interactions, interactions_device, this->targetSpheres->numElements*sizeof(bool), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(interactions_device));

  return interactions;
}
float* ParticleGraph::quantifyInteractions(){
  float* interactions = new float[this->targetSpheres->numElements];
  float* interactions_device = NULL;

  if(this->targetSpheres->state == cpu) this->targetSpheres->transferMemoryTo(gpu);
  if(this->querySpheres->state == cpu) this->querySpheres->transferMemoryTo(gpu);

  CudaSafeCall(cudaMalloc((void**)&interactions_device, this->targetSpheres->numElements*sizeof(float)));

  dim3 grid = {1,1,1};
  getGrid(this->targetSpheres->numElements,grid);
  dim3 block = {1024, 1, 1};
  interactionQuantification<<<grid,block>>>(this->targetSpheres->numElements,this->targetSpheres->device, this->querySpheres->numElements, this->querySpheres->device, interactions_device);
  CudaCheckError();

  CudaSafeCall(cudaMemcpy(interactions, interactions_device, this->targetSpheres->numElements*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(interactions_device));
  return interactions;
}


//assume all complex octrees are just protein
void ParticleGraph::determineBindingSiteTruth(ParticleList* complex){

  try{
    this->targetSpheres = complex->getSpheres(false, true, this->targetType);
    this->targetSpheres->transferMemoryTo(gpu);
    this->querySpheres = complex->getSpheres(false, false, this->queryType);
    this->querySpheres->transferMemoryTo(gpu);
  }
  catch(const LackOfTypeException  &e){
    if(this->targetSpheres != NULL){
      delete this->targetSpheres;
      this->targetSpheres = NULL;
      this->querySpheres = NULL;
      throw LackOfQueryException("lack of query type when training");
    }
    else{
      this->targetSpheres = NULL;
      this->querySpheres = NULL;
      throw LackOfTargetException("lack of target type when training");
    }
  }

  bool* interactions = checkInteractions();

  if(this->targetSpheres->state == gpu) this->targetSpheres->transferMemoryTo(cpu);

  int3 molResAtom = {-1,-1,-1};
  for(int i = 0; i < this->targetSpheres->numElements; ++i){
    molResAtom = this->targetSpheres->host[i].molResAtom;
    complex->molecules[molResAtom.x]->residues[molResAtom.y]->atoms[molResAtom.z]->truePositive = interactions[i];
  }

  this->targetSpheres->transferMemoryTo(gpu);

  delete[] interactions;
}
void ParticleGraph::fillInteractionsAndScores(ParticleList* complex, jax::Unity<bool>* &interactions, jax::Unity<float>* &scores){

  try{
    this->targetSpheres = complex->getSpheres(false, true, this->targetType);
    this->targetSpheres->transferMemoryTo(gpu);
    this->querySpheres = complex->getSpheres(false, false, this->queryType);
    this->querySpheres->transferMemoryTo(gpu);
  }
  catch(const LackOfTypeException  &e){
    if(this->targetSpheres != NULL){
      delete this->targetSpheres;
      this->targetSpheres = NULL;
      this->querySpheres = NULL;
      throw LackOfQueryException("lack of query type when training");
    }
    else{
      this->targetSpheres = NULL;
      this->querySpheres = NULL;
      throw LackOfTargetException("lack of target type when training");
    }
  }

  bool* interactions_host = checkInteractions();

  std::vector<float> atomScores;

  if(this->targetSpheres->state == gpu) this->targetSpheres->transferMemoryTo(cpu);

  int3 molResAtom = {-1,-1,-1};
  for(int i = 0; i < this->targetSpheres->numElements; ++i){
    molResAtom = this->targetSpheres->host[i].molResAtom;
    complex->molecules[molResAtom.x]->residues[molResAtom.y]->atoms[molResAtom.z]->truePositive = interactions_host[i];
    atomScores.push_back(complex->molecules[molResAtom.x]->residues[molResAtom.y]->atoms[molResAtom.z]->relativeAffinity);
  }


  float* atomScores_host = &atomScores[0];
  scores = new jax::Unity<float>(atomScores_host, this->targetSpheres->numElements, cpu);
  interactions = new jax::Unity<bool>(interactions_host, this->targetSpheres->numElements, cpu);

}


/*
GETTERS AND SETTERS
*/
void ParticleGraph::setTargetType(MoleculeType type){
  this->targetType.clear();
  this->targetType.insert(type);
}
void ParticleGraph::setTargetType(std::set<MoleculeType> types){
  this->targetType = types;
}
void ParticleGraph::setQueryType(MoleculeType type){
  this->queryType.clear();
  this->queryType.insert(type);
}
void ParticleGraph::setQueryType(std::set<MoleculeType> types){
  this->queryType = types;
}


void ParticleGraph::setConstraints(float queryEdgeConstraint, float targetEdgeConstraint, float interactionThreshold, int minInteractions){
  this->targetEdgeConstraints = {2.0f, targetEdgeConstraint};
  this->queryEdgeConstraints = {2.0f, queryEdgeConstraint};
  this->interactionThreshold = interactionThreshold;
  this->minInteractions = minInteractions;
}
void ParticleGraph::setTriangles(const std::vector<Triangle> &triangles){
  Triangle* triangles_host = new Triangle[triangles.size()];
  for(unsigned int i = 0; i < triangles.size(); ++i){
    triangles_host[i] = triangles[i];
  }
  this->triangles = new Unity<Triangle>(triangles_host, triangles.size(), cpu);
  this->triangles->transferMemoryTo(gpu);
}
std::vector<Triangle> ParticleGraph::getTriangles(){
  if(this->triangles == NULL || this->triangles->numElements == 0){
    std::cout<<"ERROR there are no triangles generated to return"<<std::endl;
    exit(-1);
  }
  std::vector<Triangle> metaTriangleVector;
  for(int i = 0; i < this->triangles->numElements; ++i){
    metaTriangleVector.push_back(this->triangles->host[i]);
  }
  return metaTriangleVector;
}
