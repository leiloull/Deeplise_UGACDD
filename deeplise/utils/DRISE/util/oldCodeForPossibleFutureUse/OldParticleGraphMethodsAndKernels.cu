//NOTE IF CALLED KERNELS OR METHODS ARE NOT HERE, THEY ARE IN PARTICLEGRAPH_CU
__global__ void computeAtomScores(const int numSphereTypes, const Sphere* targetSpheres, const int numTargetEdges,
const int2* targetEdges, const int* triangleNodes, const Triangle* triangles, float* scores){
  unsigned long globalID = getGlobalIdx_2D_1D();
  int3 triangle = {-1,-1,-1};
  int3 triangleType = {-1,-1,-1};
  if(globalID < numTargetEdges){
    int u = targetEdges[globalID].x, v = targetEdges[globalID].y;
    int u_it = triangleNodes[u], u_end = triangleNodes[u + 1];
    int v_it = triangleNodes[v], v_end = triangleNodes[v + 1];

    if((u_it == 0 && u != 0) || (v_it == 0 && v != 0)) return;

    int a = targetEdges[u_it].y, b = targetEdges[v_it].y;
    while (u_it < u_end && v_it < v_end) {
      int d = a - b;
      if (d <= 0) {
        a = targetEdges[++u_it].y;
      }
      if (d >= 0) {
        b = targetEdges[++v_it].y;
      }
      if (d == 0) {
        //is triangle between u,v,a
        triangle = {targetEdges[globalID].x,targetEdges[globalID].y,targetEdges[u_it - 1].y};
        triangleType = {targetSpheres[triangle.x].type, targetSpheres[triangle.y].type, targetSpheres[triangle.z].type};
        orderInt3(triangleType);

        unsigned long index = getTriangleTypeIndexFromOrderedList(triangleType, numSphereTypes);
        Triangle scoredTriangle = triangles[index];

        if(scoredTriangle.affinity > 0.0f){
          atomicAdd(&scores[triangle.x], scoredTriangle.affinity);
          atomicAdd(&scores[triangle.y], scoredTriangle.affinity);
          atomicAdd(&scores[triangle.z], scoredTriangle.affinity);
        }
      }
    }
  }
}
__global__ void countSurfaceOccurances(const int numTrianglePointers, const int3* trianglePointers, const Sphere* spheres, const int numTypes, int2* __restrict__ scores){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(globalID < numTrianglePointers){
    int3 trianglePointer = trianglePointers[globalID];
    Sphere targetSpheres[3] = {spheres[trianglePointer.x],spheres[trianglePointer.y],spheres[trianglePointer.z]};
    int triangleType = getTriangleTypeIndexFromOrderedList({targetSpheres[0].type,targetSpheres[1].type,targetSpheres[2].type}, numTypes);
    atomicAdd(&scores[triangleType].x, 1);
  }
}

__global__ void countInteractionOccurances(int2* __restrict__ scores, const int minInteractions, const float threshold, const int numTrianglePointers, const int3* __restrict__ trianglePointers,
int numEdges, const int2* __restrict__ edges, const Sphere* __restrict__ targetSpheres, const Sphere* __restrict__ querySpheres, const int numTypes){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  if(blockId < numTrianglePointers){
    int3 trianglePointer = trianglePointers[blockId];
    Sphere targets[3] = {targetSpheres[trianglePointer.x],targetSpheres[trianglePointer.y],targetSpheres[trianglePointer.z]};
    Sphere querys[2];
    int start = threadIdx.x;
    int stride = blockDim.x;
    int regNumEdges = numEdges;
    int2 currentEdge = {-1,-1};
    int triangleType = getTriangleTypeIndexFromOrderedList({targets[0].type,targets[1].type,targets[2].type}, numTypes);
    int regMinInteractions = minInteractions;
    float regThreshold = threshold;
    int interactionCount = 0;
    int totalInteractions = 0;
    for(int i = start; i < regNumEdges; i+=stride){
      currentEdge = edges[i];
      querys[0] = querySpheres[currentEdge.x];
      querys[1] = querySpheres[currentEdge.y];
      interactionCount = 0;
      for(int t = 0; t < 3; ++t){
        for(int q = 0; q < 2; ++q){
          if(elucid(querys[q].center, targets[t].center) < regThreshold){
            ++interactionCount;
          }
        }
      }
      if(interactionCount >= regMinInteractions){
        ++totalInteractions;
      }
    }
    __syncthreads();
    if(totalInteractions != 0){
      atomicAdd(&scores[triangleType].y, totalInteractions);
    }
  }
}
__global__ void countInteractionOccurancesSimpleThreaded(int2* __restrict__ scores, const int minInteractions, const float threshold, const int numTrianglePointers, const int3* __restrict__ trianglePointers,
int numEdges, const int2* __restrict__ edges, const Sphere* __restrict__ targetSpheres, const Sphere* __restrict__ querySpheres, const int* __restrict__ triangleReferences){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  if(blockId < numTrianglePointers){
    int3 trianglePointer = trianglePointers[blockId];
    Sphere targets[3] = {targetSpheres[trianglePointer.x],targetSpheres[trianglePointer.y],targetSpheres[trianglePointer.z]};
    Sphere querys[2];
    int start = threadIdx.x;
    int stride = blockDim.x;
    int regNumEdges = numEdges;
    int2 currentEdge = {-1,-1};
    int triangleReference = triangleReferences[blockId];
    int regMinInteractions = minInteractions;
    float regThreshold = threshold;
    int interactionCount = 0;
    int totalInteractions = 0;
    for(int i = start; i < regNumEdges; i+=stride){
      currentEdge = edges[i];
      querys[0] = querySpheres[currentEdge.x];
      querys[1] = querySpheres[currentEdge.y];
      interactionCount = 0;
      for(int t = 0; t < 3; ++t){
        for(int q = 0; q < 2; ++q){
          if(elucid(querys[q].center, targets[t].center) < regThreshold){
            ++interactionCount;
          }
        }
      }
      if(interactionCount >= regMinInteractions){
        ++totalInteractions;
      }
    }
    __syncthreads();
    if(totalInteractions != 0){
      //printf("%d\n",totalInteractions);
      atomicAdd(&scores[triangleReference].y, totalInteractions);
    }
  }
}

__global__ void edgeFinder(const int numPossible, const int numSpheres, const Sphere* spheres, int2* edges, const float2 threshold, int* edgeCounter){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(numPossible > globalID){
    float2 regThreshold = threshold;
    int2 edge = getEdgeFromIndexInOrderedList(globalID, numSpheres);
    float dist = elucid(spheres[edge.x].center,spheres[edge.y].center);
    if(dist > regThreshold.x && dist < regThreshold.y){
      edges[globalID] = edge;
      edgeCounter[globalID] = 1;
    }
  }
}

__global__ void countTrianglesAndScore(const float interactionThreshold, const int minInteractions, const int numSphereTypes,
const Sphere* targetSpheres, const Sphere* querySpheres, const int numQueryEdges, const int2* queryEdges,
const int numTargetEdges, const int2* targetEdges, const int* triangleNodes, Triangle* triangles){
  unsigned long globalID = getGlobalIdx_2D_1D();
  int3 triangle = { -1, -1, -1 };
  int3 triangleType = {-1, -1, -1};
  int2 edge = {-1,-1};
  int connections = 0;
  if(globalID < numTargetEdges){
    float regThreshold = interactionThreshold;
    int regMinInteractions = minInteractions;
    int regNumQueryEdges = numQueryEdges;
    int u = targetEdges[globalID].x, v = targetEdges[globalID].y;
    int u_it = triangleNodes[u], u_end = triangleNodes[u + 1];
    int v_it = triangleNodes[v], v_end = triangleNodes[v + 1];

    if((u_it == 0 && u != 0) ||
    (v_it == 0 && v != 0) ||
    v_end == 0 || u_end == 0) return;

    int a = targetEdges[u_it].y, b = targetEdges[v_it].y;
    while (u_it < u_end && v_it < v_end) {
      int d = a - b;
      if (d <= 0) {
        a = targetEdges[++u_it].y;
      }
      if (d >= 0) {
        b = targetEdges[++v_it].y;
      }
      if (d == 0) {
        //is triangle between u,v,a
        triangle = {targetEdges[globalID].x,targetEdges[globalID].y,targetEdges[u_it - 1].y};
        triangleType = {targetSpheres[triangle.x].type, targetSpheres[triangle.y].type, targetSpheres[triangle.z].type};
        orderInt3(triangleType);

        unsigned long index = getTriangleTypeIndexFromOrderedList(triangleType, numSphereTypes);
        //TODO need to check to see if this is correct

        atomicAdd(&triangles[index].occurances, 1);

        Sphere triangleSpheres[3] = {targetSpheres[triangle.x],targetSpheres[triangle.y],targetSpheres[triangle.z]};
        Sphere edgeSpheres[2];
        for(int qE = 0; qE < regNumQueryEdges; ++qE){
          edge = queryEdges[qE];
          connections = 0;
          edgeSpheres[0] = querySpheres[edge.x];
          edgeSpheres[1] = querySpheres[edge.y];
          for(int t = 0; t < 3; ++t){
            for(int e = 0; e < 2; ++e){
              if(elucid(triangleSpheres[t].center,edgeSpheres[e].center) <= regThreshold){
                ++connections;
              }
            }
          }
          if(regMinInteractions <= connections)atomicAdd(&triangles[index].interactions, 1);
        }
      }
    }
  }
}

void ParticleGraph::findTargetEdges_device(int2* &edges_device, int& numEdges){
  time_t start = time(nullptr);
  int possibleEdges = 1;
  for(int i = this->targetSpheres->numElements; i > this->targetSpheres->numElements - 2; --i){
    possibleEdges *= i;
  }
  possibleEdges /= 2;

  int2* edgesTemp_device;
  int* edgeCounter_device;
  CudaSafeCall(cudaMalloc((void**)&edgeCounter_device, possibleEdges*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&edgesTemp_device, possibleEdges*sizeof(int2)));
  dim3 grid;
  dim3 block;
  getFlatGridBlock(possibleEdges,grid,block);
  edgeFinder<<<grid,block>>>(possibleEdges, this->targetSpheres->numElements, (Sphere*) this->targetSpheres->device, edgesTemp_device, this->targetEdgeConstraints, edgeCounter_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<int> sumE(edgeCounter_device);
  thrust::inclusive_scan(sumE, sumE + possibleEdges, sumE);
  CudaSafeCall(cudaMemcpy(&numEdges,edgeCounter_device + (possibleEdges - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(edgeCounter_device));

  CudaSafeCall(cudaMalloc((void**)&edges_device, numEdges*sizeof(int2)));
  thrust::device_ptr<int2> toCompact(edgesTemp_device);
  thrust::device_ptr<int2> out(edges_device);
  thrust::copy_if(toCompact, toCompact + possibleEdges, out, is_real_edge());
  CudaSafeCall(cudaFree(edgesTemp_device));

  int2* edges = new int2[numEdges];
  CudaSafeCall(cudaMemcpy(edges, edges_device, numEdges*sizeof(int2), cudaMemcpyDeviceToHost));
  printf("Target: %lu spheres led to %d possibleEdges minimized to %d by constraint of %f.\n",this->targetSpheres->numElements,possibleEdges,numEdges,this->targetEdgeConstraints.y);
  std::cout << "gpu target edge finder took " << difftime(time(nullptr), start) <<" seconds."<<std::endl;
}
void ParticleGraph::findQueryEdges_device(int2* &edges_device, int& numEdges){
  int possibleEdges = 1;
  for(int i = this->querySpheres->numElements; i > this->querySpheres->numElements - 2; --i){
    possibleEdges *= i;
  }
  possibleEdges /= 2;

  int2* edgesTemp_device;
  int* edgeCounter_device;
  CudaSafeCall(cudaMalloc((void**)&edgeCounter_device, possibleEdges*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&edgesTemp_device, possibleEdges*sizeof(int2)));
  dim3 grid;
  dim3 block;
  getFlatGridBlock(possibleEdges,grid,block);
  edgeFinder<<<grid,block>>>(possibleEdges, this->querySpheres->numElements, (Sphere*) this->querySpheres->device, edgesTemp_device, this->queryEdgeConstraints, edgeCounter_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<int> sumE(edgeCounter_device);
  thrust::inclusive_scan(sumE, sumE + possibleEdges, sumE);
  CudaSafeCall(cudaMemcpy(&numEdges,edgeCounter_device + (possibleEdges - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(edgeCounter_device));

  CudaSafeCall(cudaMalloc((void**)&edges_device, numEdges*sizeof(int2)));
  thrust::device_ptr<int2> toCompact(edgesTemp_device);
  thrust::device_ptr<int2> out(edges_device);
  thrust::copy_if(toCompact, toCompact + possibleEdges, out, is_real_edge());
  CudaSafeCall(cudaFree(edgesTemp_device));
  printf("Query: %lu spheres led to %d possibleEdges minimized to %d by constraint of %f.\n",this->querySpheres->numElements,possibleEdges,numEdges,this->queryEdgeConstraints.y);
}

void ParticleGraph::trainTriangles(int numTargetEdges, int2* targetEdges_device, int numQueryEdges, int2* queryEdges_device){
  int* targetEdgeNodes_device;
  CudaSafeCall(cudaMalloc((void**)&targetEdgeNodes_device,this->targetSpheres->numElements*sizeof(int)));
  int* temp = new int[this->targetSpheres->numElements]();
  CudaSafeCall(cudaMemcpy(targetEdgeNodes_device,temp, this->targetSpheres->numElements*sizeof(int),cudaMemcpyHostToDevice));
  delete[] temp;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numTargetEdges - 1, grid, block);
  edgeNodeFinder<<<grid,block>>>(numTargetEdges, targetEdges_device, targetEdgeNodes_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(numTargetEdges, grid, block);
  countTrianglesAndScore<<<grid,block>>>(this->interactionThreshold, this->minInteractions, this->numSphereTypes,
    (Sphere*) this->targetSpheres->device, (Sphere*) this->querySpheres->device, numQueryEdges, queryEdges_device,
    numTargetEdges, targetEdges_device, targetEdgeNodes_device, this->triangles_device);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(this->triangles, this->triangles_device, this->numTriangles*sizeof(Triangle),cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(targetEdgeNodes_device));

}

void ParticleGraph::executeTriangleCounters(int numTrianglePointers, int3* trianglePointers_device, int numQueryEdges, int2* queryEdges_device){
  time_t start = time(nullptr);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numTrianglePointers, grid, block);

  int2* scores_device;
  int2* scores = new int2[this->numTriangles]();

  CudaSafeCall(cudaMalloc((void**)&scores_device, this->numTriangles*sizeof(int2)));
  CudaSafeCall(cudaMemcpy(scores_device, scores, this->numTriangles*sizeof(int2), cudaMemcpyHostToDevice));

  countSurfaceOccurances<<<grid,block>>>(numTrianglePointers, trianglePointers_device, (Sphere*) this->targetSpheres->device, this->numSphereTypes, scores_device);
  CudaCheckError();
  grid = {1,1,1};
  block = {1024,1,1};
  getGrid(numTrianglePointers, grid);

  CudaSafeCall(cudaMalloc((void**)&scores_device, this->numTriangles*sizeof(int2)));

  countInteractionOccurances<<<grid,block>>>(scores_device, this->minInteractions, this->interactionThreshold, numTrianglePointers, trianglePointers_device, numQueryEdges, queryEdges_device,
    (Sphere*) this->targetSpheres->device, (Sphere*) this->querySpheres->device, this->numSphereTypes);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(scores, scores_device, this->numTriangles*sizeof(int2), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(scores_device));

  for(int i = 0; i < this->numTriangles; ++i){
    this->triangles[i].occurances += scores[i].x;
    if(scores[i].y != 0) this->triangles[i].interactions += scores[i].y;
  }
  delete[] scores;
  CudaSafeCall(cudaMemcpy(this->triangles_device, this->triangles, this->numTriangles*sizeof(Triangle), cudaMemcpyHostToDevice));
  std::cout << "counting triangle occurances w/o references took " << difftime(time(nullptr), start) <<" seconds."<<std::endl;
}
