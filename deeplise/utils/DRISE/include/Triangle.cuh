#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "common_includes.h"
#include "bio_maps.h"

struct Triangle {
    unsigned int occurances;
    unsigned int interactions;
    float affinity;

    int3 atomTypes;

    __device__ __host__  Triangle();
    __device__ __host__ Triangle(int3 atomTypes);
    void printTriangle();
};

bool compareByOccurances(const Triangle &a, const Triangle &b);
bool compareByInteractions(const Triangle &a, const Triangle &b);
bool compareByAffinity(const Triangle &a, const Triangle &b);

#endif /* TRIANGLE_CUH */
