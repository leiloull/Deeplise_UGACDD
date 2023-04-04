#ifndef SURFACECLASSIFIER_CUH
#define SURFACECLASSIFIER_CUH

#include "common_includes.h"

#include "bio_maps.h"
#include "cuda_util.cuh"
#include "Molecule.cuh"
#include "io_util.h"
#include "Octree.cuh"
#include "ParticleList.cuh"
#include "freesasa.h"

extern __constant__ float dOffset;//doffset acting as radius of surface probe
extern __constant__ int3 halfDegrees[26];//list of 26 possible 45deg incremented directions in xyz
__device__ float square(const float &a);
//returns true if sphere and node overlap with each other
__device__ bool nodeSphereOverlapTest(const Sphere &sphere, const Node &node);
//returns true if sphere overlaps with cube defined by a center and a width
__device__ bool nodeSphereOverlapTest(const Sphere &sphere, const float3 &nodeCenter, const float &width);
//returns elucidian distance squared (distance formula without sqrt)
__device__ float elucidSq(const float3 &a, const float3 &b);
//returns 1==no overlap,0==partial overlap,1==full overlap
__device__ int nodeSphereOverlapType(const Sphere &sphere, const Node &node);
//returns 1==no overlap,0==partial overlap,1==full overlap (cube defined by center and width)
__device__ int nodeSphereOverlapType(const Sphere &sphere, const float3 &nodeCenter, const float &width);

/*
This method generates points separated equally in all nodes at a given depth.
Those points are used to perform shrake rupley on spheres that could overlap with the point.
*/
__global__ void shrakeRupleyOctree(unsigned int numNodesAtDepth, unsigned int depthStartingIndex, const Node* __restrict__ nodes, Sphere* __restrict__ spheres);
/*
This method generates points separated within spheres so that point concentration
becomes higher closer the the edge of the sphere.radius + dOffset. Shrake rupley is performed
on these points by looking at all neighboring spheres.
*/
__global__ void shrakeRupleySpheres(unsigned int numSpheres, Sphere* spheres, unsigned int* pointNodeIndex, Node* nodes, int numTypes, MoleculeType* molecularFocus);

__global__ void shrakeRupley(unsigned int numSpheres, Sphere* spheres, unsigned int* pointNodeIndex, Node* nodes, unsigned int pointDensity);


class SurfaceInterface{

  private:
      //host variables
      int3 min;
      int3 max;

      //NOTE default probe_radius = 1.4 and default dOffset = 1.4 but can be different
      freesasa_parameters params;
      float dOffset;

  public:


    time_t timer;
    SurfaceInterface();
    SurfaceInterface(float dOffset);
    SurfaceInterface(freesasa_parameters params);
    ~SurfaceInterface();


    /* custom method usage CUDA */
    void setDOffset(float dOffset);
    void extractSurface(ParticleList* complex);
    void extractSurface(ParticleList* complex, unsigned int pointDensity);
    void extractSurface(Octree* octree);
    void extractSurface(Octree* octree, unsigned int pointDensity);
    void determineTruthAndSurface(ParticleList* complex, std::set<MoleculeType> queryTypes);
    void determineTruthAndSurface(ParticleList* complex, std::set<MoleculeType> queryTypes, unsigned int pointDensity);

    /* freesasa usage */
    void setFreeSASAParams(freesasa_parameters params);
    //does solvent accessible surface area calculation for each atoms but returns total
    double extractFreeSASA(ParticleList* complex);
    double extractFreeSASA(std::vector<Molecule*> molecules);
    double extractFreeSASA(std::vector<Residue*> residues);
    double extractFreeSASA(std::vector<Atom*> atoms);
    void determineTruthAndFreeSASA(ParticleList* complex, MoleculeType targetType);
    void determineTruthAndFreeSASA(ParticleList* complex, std::set<MoleculeType> targetTypes);

};
#endif /*SURFACECLASSIFIER_CUH*/
