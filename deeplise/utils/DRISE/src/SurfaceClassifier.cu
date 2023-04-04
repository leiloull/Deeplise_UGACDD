#include "SurfaceClassifier.cuh"


__constant__ int3 halfDegrees[26] = {
  {-1,-1,-1},
  {-1,-1,0},
  {-1,-1,1},
  {-1,0,-1},
  {-1,0,0},
  {-1,0,1},
  {-1,1,-1},
  {-1,1,0},
  {-1,1,1},
  {0,-1,-1},
  {0,-1,0},
  {0,-1,1},
  {0,0,-1},
  //{0,0,0},
  {0,0,1},
  {0,1,-1},
  {0,1,0},
  {0,1,1},
  {1,-1,-1},
  {1,-1,0},
  {1,-1,1},
  {1,0,-1},
  {1,0,0},
  {1,0,1},
  {1,1,-1},
  {1,1,0},
  {1,1,1}
};
// same as above with uses index within 27 threads
// int3 line = {
//   (-1 + (dirHelper/9)),
//   (-1 + ((dirHelper%9)/3)),
//   (-1 + (dirHelper%3))
// };

__device__ float square(const float &a){
  return a*a;
}
__device__ bool nodeSphereOverlapTest(const Sphere &sphere, const Node &node, const float &dOffset){
  float distanceMin = 0;
  float3 topCorner = node.center + (node.width/2);
  float3 bottomCorner = node.center - (node.width/2);
  if(sphere.center.x < bottomCorner.x) distanceMin += square(sphere.center.x - bottomCorner.x);
  else if(sphere.center.x > topCorner.x) distanceMin += square(sphere.center.x - topCorner.x);
  if(sphere.center.y < bottomCorner.y) distanceMin += square(sphere.center.y - bottomCorner.y);
  else if(sphere.center.y > topCorner.y) distanceMin += square(sphere.center.y - topCorner.y);
  if(sphere.center.z < bottomCorner.z) distanceMin += square(sphere.center.z - bottomCorner.z);
  else if(sphere.center.z > topCorner.z) distanceMin += square(sphere.center.z - topCorner.z);
  return distanceMin <= square(sphere.radius + dOffset);
}
__device__ bool nodeSphereOverlapTest(const Sphere &sphere, const float3 &nodeCenter, const float &width, const float &dOffset){
  float distanceMin = 0;
  float3 topCorner = nodeCenter + (width/2);
  float3 bottomCorner = nodeCenter - (width/2);
  if(sphere.center.x < bottomCorner.x) distanceMin += square(sphere.center.x - bottomCorner.x);
  else if(sphere.center.x > topCorner.x) distanceMin += square(sphere.center.x - topCorner.x);
  if(sphere.center.y < bottomCorner.y) distanceMin += square(sphere.center.y - bottomCorner.y);
  else if(sphere.center.y > topCorner.y) distanceMin += square(sphere.center.y - topCorner.y);
  if(sphere.center.z < bottomCorner.z) distanceMin += square(sphere.center.z - bottomCorner.z);
  else if(sphere.center.z > topCorner.z) distanceMin += square(sphere.center.z - topCorner.z);
  return distanceMin <= square(sphere.radius + dOffset);
}
__device__ float elucidSq(const float3 &a, const float3 &b){
  return square(a.x - b.x) + square(a.y - b.y) + square(a.z - b.z);
}
__device__ int nodeSphereOverlapType(const Sphere &sphere, const Node &node, const float &dOffset){
  float distanceMin = 0;
  float3 topCorner = node.center + (node.width/2);
  float3 bottomCorner = node.center - (node.width/2);
  if(sphere.center.x < bottomCorner.x) distanceMin += square(sphere.center.x - bottomCorner.x);
  else if(sphere.center.x > topCorner.x) distanceMin += square(sphere.center.x - topCorner.x);
  if(sphere.center.y < bottomCorner.y) distanceMin += square(sphere.center.y - bottomCorner.y);
  else if(sphere.center.y > topCorner.y) distanceMin += square(sphere.center.y - topCorner.y);
  if(sphere.center.z < bottomCorner.z) distanceMin += square(sphere.center.z - bottomCorner.z);
  else if(sphere.center.z > topCorner.z) distanceMin += square(sphere.center.z - topCorner.z);

  int indicator = -1;//-1==no overlap,0==partial overlap,1==full overlap
  if(distanceMin <= square(sphere.radius + dOffset)) indicator = 0;
  float centerDist = sqrtf(square(node.center.x - sphere.center.x) + square(node.center.y - sphere.center.y) +square(node.center.z - sphere.center.z));
  if(centerDist < (sphere.radius + dOffset) - (node.width/2)){//cube is within sphere
    indicator = 1;
  }
  else if(centerDist < (node.width/2) - (sphere.radius + dOffset)){//sphere is inside cube
    indicator = 2;
  }
  return indicator;
}
__device__ int nodeSphereOverlapType(const Sphere &sphere, const float3 &nodeCenter, const float &width, const float &dOffset){
  float distanceMin = 0;
  float3 topCorner = nodeCenter + (width/2);
  float3 bottomCorner = nodeCenter - (width/2);
  if(sphere.center.x < bottomCorner.x) distanceMin += square(sphere.center.x - bottomCorner.x);
  else if(sphere.center.x > topCorner.x) distanceMin += square(sphere.center.x - topCorner.x);
  if(sphere.center.y < bottomCorner.y) distanceMin += square(sphere.center.y - bottomCorner.y);
  else if(sphere.center.y > topCorner.y) distanceMin += square(sphere.center.y - topCorner.y);
  if(sphere.center.z < bottomCorner.z) distanceMin += square(sphere.center.z - bottomCorner.z);
  else if(sphere.center.z > topCorner.z) distanceMin += square(sphere.center.z - topCorner.z);

  int indicator = -1;//-1==no overlap,0==partial overlap,1==full overlap
  if(distanceMin <= square(sphere.radius + dOffset)) indicator = 0;
  float centerDist = sqrtf(square(nodeCenter.x - sphere.center.x) + square(nodeCenter.y - sphere.center.y) +square(nodeCenter.z - sphere.center.z));
  if(centerDist < (sphere.radius + dOffset) - (width/2)){//cube is within sphere
    indicator = 1;
  }
  else if(centerDist < (width/2) - (sphere.radius + dOffset)){//sphere is inside cube
    indicator = 2;
  }
  return indicator;
}

__global__ void shrakeRupleyOctree(unsigned int numNodesAtDepth, unsigned int depthStartingIndex, const Node* __restrict__ nodes, Sphere* __restrict__ spheres, const float dOffset){
  unsigned int blockId = blockIdx.y*gridDim.x + blockIdx.x;
  if(blockId < numNodesAtDepth){//make sure blockId is less than number of nodes to be investigated
    blockId += depthStartingIndex;//jump to first node at depth of focus
    Node node = nodes[blockId];
    float3 point = node.center - (node.width/2);//instantiating point as bottom left vertex of node
    float3 interval = {node.width/blockDim.x,node.width/blockDim.y,node.width/blockDim.z};//distances away from initialized point
    float dOffset_reg = dOffset;//single access better than multiple
    //TODO remove interval variable and just put calculations into this next line
    point = {threadIdx.x*interval.x + point.x, threadIdx.y*interval.y + point.y, threadIdx.z*interval.y + point.z};
    while(node.numPoints <= 1) node = nodes[node.parent];//this travels up the octree until the center node in focus area has more than 1 sphere
    Node toSearch;
    Sphere sphere;
    float dist = 0.0f;
    int sphereIndex = -1;
    // iterate over neighbors (and itself = neighbors[13])
    for(int n = 0; n < 27; n++){
      // if node is empty
      if(node.neighbors[n] == -1 || nodes[node.neighbors[n]].numPoints == 0) continue;
      toSearch = nodes[node.neighbors[n]];
      // iterate over spheres (shrake rupley is only ever performed on spheres)
       for(int p = toSearch.pointIndex; p < toSearch.pointIndex + toSearch.numPoints; ++p){
        sphere = spheres[p];
        dist = sqrtf(elucidSq(sphere.center, point));
        //point is outside sphere
        if(dist > sphere.radius + dOffset_reg) continue;
        else if(dist < sphere.radius || sphereIndex != -1) return;//point is inside sphere or is in multiple spheres
        else{//point is inside the doffset of this sphere
          sphereIndex = p;
        }
      }
    }
    //if thread makes it here then that point was only in 1 dOffset
    if(sphereIndex != -1){
      spheres[sphereIndex].surf = true;
      //printf("%f %f %f\n",point.x,point.y,point.z);
    }
  }
}
__global__ void shrakeRupleySpheres(unsigned int numSpheres, Sphere* spheres, unsigned int* pointNodeIndex, Node* nodes, const float dOffset){
  unsigned int blockId = blockIdx.y*gridDim.x + blockIdx.x;
  if(blockId < numSpheres){
    Node node = nodes[pointNodeIndex[blockId]];//node that sphere of focus resides in (at deepest depth)
    Sphere sphere = spheres[blockId];
    __syncthreads();
    int3 direction = halfDegrees[threadIdx.x];//getting direction to distribute points on
    float dOffset_reg = dOffset;

    //this is defining the distance off the spheres radius that the point will reside
    float distOff = dOffset_reg - (dOffset_reg/powf(2.0f,threadIdx.y));//this makes points get more concentrated to doffset
    //float distOff = (dOffset_reg/blockDim.y)*((float)threadIdx.y);//this is even distribution

    float specificRadius = sphere.radius + distOff;//this variable is defining the distance from the center of the sphere to the point of focus
    float phi = acosf(direction.z/specificRadius);
    float theta = atan2f(direction.y,direction.x);
    float3 point = {sinf(phi)*cosf(theta), sinf(phi)*sinf(theta), cosf(phi)};//rotates point to reside on directional line
    point = (point*specificRadius) + sphere.center;//adds center as previous calculations had center at 0,0,0

    //traverse octree upward until node at the center of focus has more than one point
    //this is done to ensure search space is large enough to include possibly overlapping spheres
    while(node.numPoints <= 2) node = nodes[node.parent];
    Node neighbor;
    float dist = 0.0f;
    //iterate through search space

    for(int n = 0; n < 27; ++n){
      if(node.neighbors[n] == -1 || nodes[node.neighbors[n]].numPoints == 0) continue;
      neighbor = nodes[node.neighbors[n]];
      for(int p = neighbor.pointIndex; p < neighbor.numPoints + neighbor.pointIndex; ++p){
        sphere = spheres[p];
        dist = sqrtf(elucidSq(sphere.center, point));
        if(dist > sphere.radius + dOffset_reg) continue;
        else if(dist < sphere.radius || p != blockId) return;
      }
    }
    spheres[blockId].surf = true;//if thread makes it here this sphere is on the surface
  }
}
__global__ void shrakeRupley(unsigned int numSpheres, Sphere* spheres, unsigned int* pointNodeIndex, Node* nodes, unsigned int pointDensity, const float dOffset){
  unsigned int blockId = blockIdx.y*gridDim.x + blockIdx.x;
  if(blockId < numSpheres){
    Node node = nodes[pointNodeIndex[blockId]];//node that sphere of focus resides in (at deepest depth)
    Sphere sphere = spheres[blockId];
    //TODO FIX PROBLEM WITH OCTREE
    // if(sqrtf(elucidSq(node.center, sphere.center)) > node.width/2.0f){
      // printf("%f\n",sqrtf(elucidSq(node.center, sphere.center))-node.width/2.0f);
      // bool found = false;
      // for(int s = node.pointIndex; s < node.numPoints + node.pointIndex; ++s){
        // if(s == blockId){
          // found = true;
        // }
      // }
      // if(!found){
        // asm("trap;");
      // }
    // }
    float dOffset_reg = dOffset;

    //define bounding box of search area
    float internalCubeHalfWidth = 0.577350269f*sphere.radius;
    float voxelWidth = (dOffset_reg + sphere.radius) - internalCubeHalfWidth;
    int numShrakePointsPerDimension = ceil(pointDensity*voxelWidth);
    float distanceBetweenPoints = voxelWidth/(float) numShrakePointsPerDimension;

    float3 voxelMin = halfDegrees[threadIdx.x]*internalCubeHalfWidth;
    if(voxelMin.x < 0.0f) voxelMin.x -= voxelWidth;
    else if(voxelMin.x == 0.0f) voxelMin.x -= voxelWidth/2;
    if(voxelMin.y < 0.0f) voxelMin.y -= voxelWidth;
    else if(voxelMin.y == 0.0f) voxelMin.y -= voxelWidth/2;
    if(voxelMin.z < 0.0f) voxelMin.z -= voxelWidth;
    else if(voxelMin.z == 0.0f) voxelMin.z -= voxelWidth/2;
    voxelMin = voxelMin + sphere.center;

    while(node.numPoints <= 3) node = nodes[node.parent];
    Node toSearch;
    float distSq = 0.0f;
    float3 shrakePoint;
    bool alive = true;
    for(int x = threadIdx.y; x < numShrakePointsPerDimension; x+=blockDim.y){
      for(int y = threadIdx.z; y < numShrakePointsPerDimension; y+=blockDim.z){
        for(int z = 0; z < numShrakePointsPerDimension; ++z){
          shrakePoint = {(float)x,(float)y,(float)z};
          shrakePoint = (shrakePoint*distanceBetweenPoints) + voxelMin;
          alive = true;
          for(int n = 0; n < 27 && alive; ++n){
            if(node.neighbors[n] == -1 || nodes[node.neighbors[n]].numPoints == 0) continue;
            toSearch = nodes[node.neighbors[n]];
            for(int s = toSearch.pointIndex; s < toSearch.numPoints + toSearch.pointIndex; ++s){
              if(s == blockId) continue;
              sphere = spheres[s];
              distSq = elucidSq(sphere.center, shrakePoint);
              if(distSq < (sphere.radius + dOffset_reg)*(sphere.radius + dOffset_reg)){
                alive = false;
                break;
              }
            }
          }
          if(alive){
            //printf("%f %f %f\n",shrakePoint.x,shrakePoint.y,shrakePoint.z);
            spheres[blockId].surf = true;
            return;
          }
        }
      }
    }
  }
}

SurfaceInterface::SurfaceInterface(){
  this->params = {FREESASA_SHRAKE_RUPLEY, 1.4, 100, 20, 1};
  this->dOffset = 1.4;
}

SurfaceInterface::SurfaceInterface(float dOffset){
  this->dOffset = dOffset;
  this->params = {FREESASA_SHRAKE_RUPLEY, 1.4, 100, 20, 1};
}
SurfaceInterface::SurfaceInterface(freesasa_parameters params){
  this->dOffset = 1.4;
  this->params = params;
}
SurfaceInterface::~SurfaceInterface(){

}

void SurfaceInterface::setDOffset(float dOffset){
  this->dOffset = dOffset;
}

void SurfaceInterface::extractSurface(ParticleList* complex){
  std::cout<<"---------------------Surface-----------------------" <<std::endl;
  time_t startTime = time(nullptr);
  MemoryState origin[3];
  if(complex->octree == NULL){
    throw ISEException_runtime("NULL OCTREE");
  }
  if(complex->octree->nodes != NULL){
    origin[0] = complex->octree->nodes->state;
    origin[1] = complex->octree->nodeDepthIndex->state;
    complex->octree->nodes->transferMemoryTo(gpu);
    complex->octree->nodeDepthIndex->transferMemoryTo(cpu);
  }
  else{
    throw NullUnityException("complex->octree->nodes");
  }
  if(complex->octree->spheres != NULL){
    origin[2] = complex->octree->spheres->state;
    complex->octree->spheres->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->spheres (spheres)");
  }

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  Node* nodes_device = complex->octree->nodes->device;
  Sphere* spheres_device = complex->octree->spheres->device;
  unsigned int numSpheres = complex->octree->spheres->numElements;
  getGrid(numSpheres, grid);

  unsigned int* nodeDepthIndex_host = complex->octree->nodeDepthIndex->host;
  unsigned int numNodesAtDepth = 1;

  grid = {1,1,1};
  block = {8,8,8};//x,y,z # of points in shrakeRupleyOctree
  int startingDepth = 0;

  while(complex->octree->width/pow(2,startingDepth) > 8.0f) startingDepth++; //TODO: edit for better startingdepth
  for(int d = startingDepth - 1; d < complex->octree->depth; ++d){
    numNodesAtDepth = nodeDepthIndex_host[complex->octree->depth - d + 1] - nodeDepthIndex_host[complex->octree->depth - d];
    getGrid(numNodesAtDepth, grid);
    shrakeRupleyOctree<<<grid,block>>>(numNodesAtDepth, nodeDepthIndex_host[complex->octree->depth - d], nodes_device, spheres_device, this->dOffset);
    CudaCheckError();
  }

  //TODO check if this updates spheres withing atoms
  complex->octree->spheres->transferMemoryTo(cpu);
  Sphere* spheres_host = complex->octree->spheres->host;
  unsigned int numSurfaceFound  = 0;
  for(int i = 0; i < complex->octree->spheres->numElements; ++i){
    if(spheres_host[i].surf){
      complex->molecules[spheres_host[i].molResAtom.x]->residues[spheres_host[i].molResAtom.y]->atoms[spheres_host[i].molResAtom.z]->sphere.surf = true;
      numSurfaceFound++;
    }
  }

  std::cout<<"NUM SURFACE FOUND = "<<numSurfaceFound<<std::endl;

  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for solvent accessible surface determination."<<std::endl;

  complex->octree->nodes->transferMemoryTo(origin[0]);
  complex->octree->nodeDepthIndex->transferMemoryTo(origin[1]);
  complex->octree->spheres->transferMemoryTo(origin[2]);
}
void SurfaceInterface::extractSurface(ParticleList* complex, unsigned int pointDensity){
  std::cout<<"---------------------Surface-----------------------" <<std::endl;
  if(pointDensity > 10){
    std::cout<<"ERROR cannot have point density higher than 10"<<std::endl;
    exit(10);
  }
  time_t startTime = time(nullptr);
  MemoryState origin[4];
  if(complex->octree == NULL){
    throw ISEException_runtime("NULL OCTREE");
  }
  // Check to make sure octree was initialized, allocate memory
  if(complex->octree->nodes != NULL){
    origin[0] = complex->octree->nodes->state;
    complex->octree->nodes->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->nodes");
  }
  if(complex->octree->spheres != NULL){
    origin[1] = complex->octree->spheres->state;
    complex->octree->spheres->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->spheres (spheres)");
  }
  if(complex->octree->pointNodeIndex != NULL){
    origin[2] = complex->octree->pointNodeIndex->state;
    complex->octree->pointNodeIndex->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->pointNodeIndex");
  }

  Node* nodes_device = complex->octree->nodes->device;
  Sphere* spheres_device = complex->octree->spheres->device;
  unsigned int* pointNodeIndex_device = complex->octree->pointNodeIndex->device;
  unsigned int numSpheres = complex->octree->spheres->numElements;

  dim3 grid = {1,1,1};
  getGrid(numSpheres, grid);
  //max of radius TODO change to that dynamically
  dim3 block = {26,1,1};

  int mostPointsInVoxel = ceil(3.03+this->dOffset)*pointDensity;
  if(mostPointsInVoxel*26 < 1024){
    block.y = mostPointsInVoxel;
    while(26*mostPointsInVoxel*(1+block.z) < 1024) ++block.z;
  }
  else{
    block.y = floor(1024/26);
  }

  getGrid(numSpheres, grid);
  shrakeRupley<<<grid,block>>>(numSpheres, spheres_device, pointNodeIndex_device, nodes_device, pointDensity, this->dOffset);
  CudaCheckError();

  //TODO check if this updates spheres withing atoms
  complex->octree->spheres->transferMemoryTo(cpu);
  Sphere* spheres_host = complex->octree->spheres->host;
  unsigned int numSurfaceFound  = 0;
  for(int i = 0; i < complex->octree->spheres->numElements; ++i){
    if(spheres_host[i].surf){
      complex->molecules[spheres_host[i].molResAtom.x]->residues[spheres_host[i].molResAtom.y]->atoms[spheres_host[i].molResAtom.z]->sphere.surf = true;
      numSurfaceFound++;
    }
  }


  std::cout<<"NUM SURFACE FOUND = "<<numSurfaceFound<<std::endl;

  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for solvent accessible surface determination."<<std::endl;

  complex->octree->nodes->transferMemoryTo(origin[0]);
  if(origin[0] == cpu) complex->octree->nodes->clear(gpu);
  complex->octree->spheres->transferMemoryTo(origin[1]);
  if(origin[1] == cpu) complex->octree->spheres->clear(gpu);
  complex->octree->pointNodeIndex->transferMemoryTo(origin[2]);
  if(origin[2] == cpu) complex->octree->pointNodeIndex->clear(gpu);
}
void SurfaceInterface::extractSurface(Octree* octree){
  std::cout<<"---------------------Surface-----------------------" <<std::endl;
  time_t startTime = time(nullptr);
  MemoryState origin[4];
  if(octree == NULL){
    throw ISEException_runtime("NULL OCTREE");
  }
  // Check to make sure octree was initialized, allocate memory
  if(octree->nodes != NULL){
    origin[0] = octree->nodes->state;
    origin[1] = octree->nodeDepthIndex->state;
    octree->nodes->transferMemoryTo(gpu);
    octree->nodeDepthIndex->transferMemoryTo(cpu);
  }
  else{
    throw NullUnityException("octree->nodes");
  }
  if(octree->spheres != NULL){
    origin[2] = octree->spheres->state;
    octree->spheres->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("octree->spheres (spheres)");
  }

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};


  Node* nodes_device = octree->nodes->device;
  Sphere* spheres_device = octree->spheres->device;
  unsigned int numSpheres = octree->spheres->numElements;
  getGrid(numSpheres, grid);

  unsigned int* nodeDepthIndex_host = octree->nodeDepthIndex->host;
  unsigned int numNodesAtDepth = 1;

  grid = {1,1,1};
  block = {8,8,8};//x,y,z # of points in shrakeRupleyOctree
  int startingDepth = 0;

  while(octree->width/pow(2,startingDepth) > 8.0f) startingDepth++; //TODO: edit for better startingdepth

  for(int d = startingDepth - 1; d < octree->depth; ++d){
    numNodesAtDepth = nodeDepthIndex_host[octree->depth - d + 1] - nodeDepthIndex_host[octree->depth - d];
    getGrid(numNodesAtDepth, grid);
    shrakeRupleyOctree<<<grid,block>>>(numNodesAtDepth, nodeDepthIndex_host[octree->depth - d], nodes_device, spheres_device, this->dOffset);
    CudaCheckError();
  }

  //TODO check if this updates spheres withing atoms
  octree->spheres->transferMemoryTo(cpu);
  Sphere* spheres_host = octree->spheres->host;
  unsigned int numSurfaceFound  = 0;
  for(int i = 0; i < octree->spheres->numElements; ++i){
    if(spheres_host[i].surf){
      numSurfaceFound++;
    }
  }

  std::cout<<"NUM SURFACE FOUND = "<<numSurfaceFound<<std::endl;

  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for solvent accessible surface determination."<<std::endl;

  octree->nodes->transferMemoryTo(origin[0]);
  if(origin[0] == cpu) octree->nodes->clear(gpu);
  octree->spheres->transferMemoryTo(origin[2]);
  if(origin[2] == cpu) octree->spheres->clear(gpu);
  octree->nodeDepthIndex->transferMemoryTo(origin[2]);
  if(origin[1] == gpu) octree->nodeDepthIndex->clear(cpu);
}
void SurfaceInterface::extractSurface(Octree* octree, unsigned int pointDensity){
  std::cout<<"---------------------Surface-----------------------" <<std::endl;
  if(pointDensity > 10){
    std::cout<<"ERROR cannot have point density higher than 10"<<std::endl;
    exit(10);
  }
  time_t startTime = time(nullptr);
  MemoryState origin[4];
  if(octree == NULL){
    throw ISEException_runtime("NULL OCTREE");
  }
  // Check to make sure octree was initialized, allocate memory
  if(octree->nodes != NULL){
    origin[0] = octree->nodes->state;
    octree->nodes->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->nodes");
  }
  if(octree->spheres != NULL){
    origin[1] = octree->spheres->state;
    octree->spheres->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->spheres (spheres)");
  }
  if(octree->pointNodeIndex != NULL){
    origin[2] = octree->pointNodeIndex->state;
    octree->pointNodeIndex->transferMemoryTo(gpu);
  }
  else{
    throw NullUnityException("complex->octree->pointNodeIndex");
  }

  Node* nodes_device = octree->nodes->device;
  Sphere* spheres_device = octree->spheres->device;
  unsigned int* pointNodeIndex_device = octree->pointNodeIndex->device;
  unsigned int numSpheres = octree->spheres->numElements;

  dim3 grid = {1,1,1};
  getGrid(numSpheres, grid);
  //max of radius TODO change to that dynamically
  dim3 block = {26,1,1};

  int mostPointsInVoxel = ceil(3.03+this->dOffset)*pointDensity;
  if(mostPointsInVoxel*26 < 1024){
    block.y = mostPointsInVoxel;
    while(26*mostPointsInVoxel*(1+block.z) < 1024) ++block.z;
  }
  else{
    block.y = floor(1024/26);
  }

  getGrid(numSpheres, grid);
  shrakeRupley<<<grid,block>>>(numSpheres, spheres_device, pointNodeIndex_device, nodes_device, pointDensity, this->dOffset);
  CudaCheckError();

  //TODO check if this updates spheres withing atoms
  octree->spheres->transferMemoryTo(cpu);
  unsigned int numSurfaceFound  = 0;
  for(int i = 0; i < octree->spheres->numElements; ++i){
    if(octree->spheres->host[i].surf){
      numSurfaceFound++;
    }
  }

  std::cout<<"NUM SURFACE FOUND = "<<numSurfaceFound<<std::endl;

  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for solvent accessible surface determination."<<std::endl;

  octree->nodes->transferMemoryTo(origin[0]);
  if(origin[0] == cpu) octree->nodes->clear(gpu);
  octree->spheres->transferMemoryTo(origin[1]);
  if(origin[1] == cpu) octree->spheres->clear(gpu);
  octree->pointNodeIndex->transferMemoryTo(origin[2]);
  if(origin[2] == cpu) octree->pointNodeIndex->clear(gpu);
}
void SurfaceInterface::determineTruthAndSurface(ParticleList* complex, std::set<MoleculeType> queryTypes){

  if(complex->octree == NULL){
    throw ISEException_runtime("MUST HAVE OCTREE SET WITH TARGET PROTEIN BEFORE DOUBLE SURFACE TRUTH DETERMINATION");
  }

  std::set<MoleculeType> allTypes;
  for(auto type = complex->typesInOctree.begin(); type != complex->typesInOctree.end(); ++type){
    if(queryTypes.find((*type)) != queryTypes.end()){
      std::cout<<"ERROR query types should not be in complex octree when attempting to determine truth through SASA"<<std::endl;
      exit(-1);
    }
    allTypes.insert((*type));
  }
  allTypes.insert(queryTypes.begin(), queryTypes.end());

  this->extractSurface(complex);

  Octree* fullOctree = complex->createOctree(false, false, false, allTypes);
  for(int i = 0; i < fullOctree->spheres->numElements; ++i){
    fullOctree->spheres->host[i].surf = false;
  }

  this->extractSurface(fullOctree);

  fullOctree->spheres->transferMemoryTo(cpu);

  int numTruth = 0;
  for(int i = 0; i < fullOctree->spheres->numElements; ++i){
    if(complex->typesInOctree.find(fullOctree->spheres->host[i].mol_type) != complex->typesInOctree.end()){
      if(!fullOctree->spheres->host[i].surf &&
      complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->sphere.surf){
        complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->truePositive = true;
        numTruth++;
      }
      else{//not truth
        complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->truePositive = false;
      }
    }
    else{//not focused type
      complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->truePositive = false;
    }
  }
  if(numTruth == 0){
    std::cout<<"ERROR NO TRUTH DETECTED"<<std::endl;
    //exit(-1);
  }
  else{
    std::cout<<"NUM TRUTH = "<<numTruth<<std::endl;
  }
  delete fullOctree;

}
void SurfaceInterface::determineTruthAndSurface(ParticleList* complex, std::set<MoleculeType> queryTypes, unsigned int pointDensity){

  if(complex->octree == NULL){
    throw ISEException_runtime("MUST HAVE OCTREE SET WITH TARGET PROTEIN BEFORE DOUBLE SURFACE TRUTH DETERMINATION");
  }

  std::set<MoleculeType> allTypes;
  for(auto type = complex->typesInOctree.begin(); type != complex->typesInOctree.end(); ++type){
    if(queryTypes.find((*type)) != queryTypes.end()){
      std::cout<<"ERROR query types should not be in complex octree when attempting to determine truth through SASA"<<std::endl;
      exit(-1);
    }
    allTypes.insert((*type));
  }
  allTypes.insert(queryTypes.begin(), queryTypes.end());

  this->extractSurface(complex, pointDensity);

  Octree* fullOctree = complex->createOctree(false, false, false, allTypes);
  for(int i = 0; i < fullOctree->spheres->numElements; ++i){
    fullOctree->spheres->host[i].surf = false;
  }

  this->extractSurface(fullOctree, pointDensity);

  fullOctree->spheres->transferMemoryTo(cpu);

  int numTruth = 0;
  for(int i = 0; i < fullOctree->spheres->numElements; ++i){
    if(complex->typesInOctree.find(fullOctree->spheres->host[i].mol_type) != complex->typesInOctree.end()){
      if(!fullOctree->spheres->host[i].surf &&
      complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->sphere.surf){
        complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->truePositive = true;
        numTruth++;
      }
      else{//not truth
        complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->truePositive = false;
      }
    }
    else{//not focused type
      complex->molecules[fullOctree->spheres->host[i].molResAtom.x]->residues[fullOctree->spheres->host[i].molResAtom.y]->atoms[fullOctree->spheres->host[i].molResAtom.z]->truePositive = false;
    }
  }
  if(numTruth == 0){
    std::cout<<"ERROR NO TRUTH DETECTED"<<std::endl;
    //exit(-1);
  }
  else{
    std::cout<<"NUM TRUTH = "<<numTruth<<std::endl;
  }
  delete fullOctree;

}

void SurfaceInterface::setFreeSASAParams(freesasa_parameters params){
  this->params = params;
}
double SurfaceInterface::extractFreeSASA(ParticleList* complex){
  std::cout<<"-----------------FreeSASASurface-------------------" <<std::endl;
  time_t startTime = time(nullptr);

  std::vector<Sphere> allAtoms;

  for(auto mol = complex->molecules.begin(); mol != complex->molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        allAtoms.push_back((*atom)->sphere);
      }
    }
  }

  double* freesasa_xyz = new double[allAtoms.size()*3];
  double* freesasa_radii = new double[allAtoms.size()];
  for(int atom = 0; atom < allAtoms.size(); ++atom){
    freesasa_radii[atom] = allAtoms[atom].radius;
    freesasa_xyz[atom*3] = allAtoms[atom].center.x;
    freesasa_xyz[atom*3 + 1] = allAtoms[atom].center.y;
    freesasa_xyz[atom*3 + 2] = allAtoms[atom].center.z;
  }
  freesasa_result* result = freesasa_calc_coord(freesasa_xyz,freesasa_radii,allAtoms.size(),&this->params);
  delete[] freesasa_xyz;
  delete[] freesasa_radii;

  double totalArea = result->total;

  int i = 0;
  unsigned int numSurface = 0;
  for(auto mol = complex->molecules.begin(); mol != complex->molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom, ++i){
        if(result->sasa[i] > 0.0){
          (*atom)->sphere.surf = true;
          ++numSurface;
        }
        else{
          (*atom)->sphere.surf = false;
        }
      }
    }
  }
  std::cout<<"NUM SURFACE FOUND "<<numSurface<<std::endl;
  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for freesasa determination."<<std::endl;
  freesasa_result_free(result);
  return totalArea;
}
double SurfaceInterface::extractFreeSASA(std::vector<Molecule*> molecules){
  std::cout<<"-----------------FreeSASASurface-------------------" <<std::endl;
  time_t startTime = time(nullptr);
  std::vector<Sphere> allAtoms;

  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        allAtoms.push_back((*atom)->sphere);
      }
    }
  }

  double* freesasa_xyz = new double[allAtoms.size()*3];
  double* freesasa_radii = new double[allAtoms.size()];
  for(int atom = 0; atom < allAtoms.size(); ++atom){
    freesasa_radii[atom] = allAtoms[atom].radius;
    freesasa_xyz[atom*3] = allAtoms[atom].center.x;
    freesasa_xyz[atom*3 + 1] = allAtoms[atom].center.y;
    freesasa_xyz[atom*3 + 2] = allAtoms[atom].center.z;
  }

  freesasa_result* result = freesasa_calc_coord(freesasa_xyz,freesasa_radii,allAtoms.size(),&this->params);
  delete[] freesasa_xyz;
  delete[] freesasa_radii;
  double totalArea = result->total;

  int i = 0;
  unsigned int numSurface = 0;
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom, ++i){
        if(result->sasa[i] > 0.0){
          (*atom)->sphere.surf = true;
          ++numSurface;
        }
        else{
          (*atom)->sphere.surf = false;
        }
      }
    }
  }
  std::cout<<"NUM SURFACE FOUND "<<numSurface<<std::endl;
  freesasa_result_free(result);
  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for freesasa determination."<<std::endl;
  return totalArea;
}
double SurfaceInterface::extractFreeSASA(std::vector<Residue*> residues){
  std::cout<<"-----------------FreeSASASurface-------------------" <<std::endl;
  time_t startTime = time(nullptr);
  std::vector<Sphere> allAtoms;

  for(auto res = residues.begin(); res != residues.end(); ++res){
    for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
      allAtoms.push_back((*atom)->sphere);
    }
  }


  double* freesasa_xyz = new double[allAtoms.size()*3];
  double* freesasa_radii = new double[allAtoms.size()];
  for(int atom = 0; atom < allAtoms.size(); ++atom){
    freesasa_radii[atom] = allAtoms[atom].radius;
    freesasa_xyz[atom*3] = allAtoms[atom].center.x;
    freesasa_xyz[atom*3 + 1] = allAtoms[atom].center.y;
    freesasa_xyz[atom*3 + 2] = allAtoms[atom].center.z;

  }

  freesasa_result* result = freesasa_calc_coord(freesasa_xyz,freesasa_radii,allAtoms.size(),&this->params);
  delete[] freesasa_xyz;
  delete[] freesasa_radii;

  double totalArea = result->total;

  int i = 0;
  unsigned int numSurface = 0;
  for(auto res = residues.begin(); res != residues.end(); ++res){
    for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom, ++i){
      if(result->sasa[i] > 0.0){
        (*atom)->sphere.surf = true;
        ++numSurface;
      }
      else{
        (*atom)->sphere.surf = false;
      }
    }
  }

  std::cout<<"NUM SURFACE FOUND "<<numSurface<<std::endl;

  freesasa_result_free(result);
  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for freesasa determination."<<std::endl;
  return totalArea;
}
double SurfaceInterface::extractFreeSASA(std::vector<Atom*> atoms){
  std::cout<<"-----------------FreeSASASurface-------------------" <<std::endl;
  time_t startTime = time(nullptr);
  double* freesasa_xyz = new double[atoms.size()*3];
  double* freesasa_radii = new double[atoms.size()];

  int i = 0;
  for(auto atom = atoms.begin(); atom != atoms.end(); ++atom, ++i){
    freesasa_xyz[i*3] = (*atom)->sphere.center.x;
    freesasa_xyz[i*3 + 1] = (*atom)->sphere.center.y;
    freesasa_xyz[i*3 + 2] = (*atom)->sphere.center.z;
    freesasa_radii[i] = (*atom)->sphere.radius;
  }

  freesasa_result* result = freesasa_calc_coord(freesasa_xyz, freesasa_radii, atoms.size(), &this->params);
  delete[] freesasa_xyz;
  delete[] freesasa_radii;

  double totalArea = result->total;

  i = 0;
  unsigned int numSurface = 0;
  for(auto atom = atoms.begin(); atom != atoms.end(); ++atom, ++i){
    if(result->sasa[i] > 0.0){
      (*atom)->sphere.surf = true;
      numSurface++;
    }
    else{
      (*atom)->sphere.surf = false;
    }
  }
  std::cout<<"NUM SURFACE FOUND "<<numSurface<<std::endl;

  freesasa_result_free(result);
  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for freesasa determination."<<std::endl;
  return totalArea;
}
void SurfaceInterface::determineTruthAndFreeSASA(ParticleList* complex, MoleculeType targetType){

  this->extractFreeSASA(complex);

  std::vector<bool> previousResults;
  std::vector<Atom*> target = complex->getAtoms(targetType);
  for(auto atom = target.begin(); atom != target.end(); ++atom){
    previousResults.push_back((*atom)->sphere.surf);
  }

  this->extractFreeSASA(target);
  int numTruth = 0;
  for(int i = 0; i < target.size(); ++i){
    if(target[i]->sphere.surf && !previousResults[i]){
      target[i]->truePositive = true;
      ++numTruth;
    }
    else{
      target[i]->truePositive = false;
    }
  }
  if(numTruth == 0){
    std::cout<<"ERROR NO TRUTH DETECTED"<<std::endl;
    //exit(-1);
  }
  else{
    std::cout<<"NUM TRUTH = "<<numTruth<<std::endl;
  }
}
void SurfaceInterface::determineTruthAndFreeSASA(ParticleList* complex, std::set<MoleculeType> targetTypes){

  this->extractFreeSASA(complex);

  std::vector<bool> previousResults;
  std::vector<Atom*> target = complex->getAtoms(targetTypes);
  for(auto atom = target.begin(); atom != target.end(); ++atom){
    previousResults.push_back((*atom)->sphere.surf);
  }

  this->extractFreeSASA(target);
  int numTruth = 0;
  for(int i = 0; i < target.size(); ++i){
    if(target[i]->sphere.surf && previousResults[i]){
      target[i]->truePositive = true;
      ++numTruth;
    }
    else{
      target[i]->truePositive = false;
    }
  }
  if(numTruth == 0){
    std::cout<<"ERROR NO TRUTH DETECTED"<<std::endl;
    //exit(-1);
  }
  else{
    std::cout<<"NUM TRUTH = "<<numTruth<<std::endl;
  }
}
