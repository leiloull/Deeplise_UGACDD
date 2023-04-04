#ifndef PARTICLEGRAPH_CUH
#define PARTICLEGRAPH_CUH

#include "common_includes.h"
#include "bio_maps.h"
#include "cuda_util.cuh"
#include "Molecule.cuh"
#include "Triangle.cuh"
#include "Octree.cuh"
#include "Unity.cuh"
#include "ParticleList.cuh"

struct is_real_edge{
  __host__ __device__
  bool operator()(const int2 x){
    return x.y != 0;
  }
};
struct is_real_triangle{
  __host__ __device__
  bool operator()(const int3 x){
    return x.y != 0;
  }
};
struct is_real_connection{
  __host__ __device__
  bool operator()(const int2 x){
    return x.x != -1;
  }
};
__device__ __forceinline__ float atomicMinFloat (float * addr, float value);
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
__device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();
__device__ __host__ __forceinline__ float square(const float &a);
__device__ __host__ __forceinline__ float elucid(const float3 &a, const float3 &b);
__device__ int getEdgeIndexFromOrderedList(const int2 &edge, const int &numElements);
__device__ int getTriangleIndexFromOrderedList(const int3 &triangle, const int &numElements);
__device__ int getTriangleTypeIndexFromOrderedList(const int3 &triangle, const int &numTypes);
__device__ int2 getEdgeFromIndexInOrderedList(const int &index, const int &numElements);
__device__ __forceinline__ void orderInt3(int3 &toOrder);

//TODO make triangle pointers int4 (pointer,pointer,pointer,ref)

__global__ void edgeNodeFinder(const int numEdges, const int2* edges, int* edgeNodes);

__global__ void countTriangles(const int numSpheres, const int numEdges, const int2* edges, const int* triangleNodes, int* numTriangles);
__global__ void recordTriangles(const int numSpheres, const int numEdges, const int2* edges, const int* triangleNodes, int3* triangles, int* index);

__global__ void triangleRefFinder(const int numTrianglePointers, const int3* __restrict__ trianglePointers, const Sphere* __restrict__ spheres ,const int numSphereTypes, int* __restrict__ triangleReferences);

__global__ void countSurfaceOccurances(const int numTrianglePointers, const int* __restrict__ triangleReferences, int2* __restrict__ scores);

__global__ void countInteractionOccurances(int2* __restrict__ scores, const int minInteractions, const float threshold, const int numTrianglePointers, const int3* __restrict__ trianglePointers,
int numEdges, const int2* __restrict__ edges, const Sphere* __restrict__ targetSpheres, const Sphere* __restrict__ querySpheres, const int* __restrict__ triangleReferences);


//verification and adjacency
__global__ void interactionQuantification(unsigned int numTargetSpheres, Sphere* targetSpheres, unsigned int numQuerySpheres, Sphere* querySpheres, float* interactions);
__global__ void interactionQuantification(unsigned int numTargetSpheres, Sphere* targetSpheres, unsigned int numQuerySpheres, Sphere* querySpheres, float* interactions);

__global__ void computeAdjacency(unsigned int numSpheres, unsigned int maxEdges, int* rows, int* columns,
Sphere* spheres);

__global__ void normalizer(const int numTriangles, Triangle* triangles, uint2 sums);
__global__ void computeAtomScores(const int numTrianglePointers, const int* triangleReferences,
  const int3* trianglePointers, const Triangle* triangles, float* scores, int* triangleCounter);

class ParticleGraph{

private:

  jax::Unity<Sphere>* targetSpheres;
  std::set<MoleculeType> targetType;
  jax::Unity<Sphere>* querySpheres;
  std::set<MoleculeType> queryType;

  int numSphereTypes;
  Unity<Triangle>* triangles;

  void generateTargetTriangleReferences(int numTrianglePointers, int3* trianglePointers_device, int* &triangleReferences_device);
  void generateTargetTriangles(int numEdges, int2* edges_device, int3* &trianglePointers_device, int &numTriangles);
  void findTargetEdges(int2* &edges, int& numEdges);
  void findQueryEdges(int2* &edges, int& numEdges);
  void executeTriangleCounters(int numTrianglePointers, int* triangleReferences_device, int3* trianglePointers_device, int numQueryEdges, int2* queryEdges_device);

  //OPTIMIZE
  void updateTrianglesFromSpheres();

  bool* checkInteractions();
  float* quantifyInteractions();

public:

  float2 edgeConstraints;//target,query

  float2 targetEdgeConstraints;
  float2 queryEdgeConstraints;

  float interactionThreshold;
  int minInteractions;

  /*
  CONSTRUCTORS AND DESTRUCTORS
  */
  ParticleGraph();
  ParticleGraph(float edgeConstraint);
  ParticleGraph(float2 edgeConstraints);
  ParticleGraph(float2 edgeConstraints, float interactionThreshold, int minInteractions);
  ParticleGraph(float2 targetEdgeConstraints, float2 queryEdgeConstraints, float interactionThreshold, int minInteractions);
  ~ParticleGraph();

  //TODO allow user to specify in method argument if this is to be executed on cpu or gpu
  void buildParticleGraph(ParticleList* complex);
  void normalizeTriangles();
  void normalizeTriangles(float(*manipulator)(float));
  void updateScores(ParticleList* complex);


  //TO IMPLEMENT
  void determineBindingSiteTruth(ParticleList* complex);
  void fillInteractionsAndScores(ParticleList* complex, jax::Unity<bool>* &interactions, jax::Unity<float>* &scores);

  /*
  GETTERS AND SETTERS
  */
  void setTargetType(MoleculeType type);
  void setTargetType(std::set<MoleculeType> types);
  void setQueryType(MoleculeType type);
  void setQueryType(std::set<MoleculeType> types);

  void setConstraints(float queryEdgeConstraint, float targetEdgeConstraint, float interactionThreshold, int minInteractions);
  void setTriangles(const std::vector<Triangle> &triangles);
  std::vector<Triangle> getTriangles();
};


#endif /* PARTICLEGRAPH_CUH */
