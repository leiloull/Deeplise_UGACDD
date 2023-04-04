#ifndef COMMON_INCLUDES_H
#define COMMON_INCLUDES_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdlib>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <cstring>
#include <set>
#include <map>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <memory>
#include <dirent.h>
#include <iterator>
#include <cfloat>
#include "ISEExceptions.hpp"
#include <set>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "Unity.cuh"
using namespace jax;

#define ATOMTYPE 0
#define SURFACE 1
#define TRAIN 2
#define BSP 3
#define TRAINWOSURF 4
#define BSPWOSURF 5

typedef enum ConnectorType{
  file_io = 0,
  local_database = 1,
  remote_database = 2
} ConnectorType;

typedef enum FileType{
  PDB,
  JSON,
  PDBQT,
  SDF,
  MOL2,
  NO_FILE
} FileType;

typedef enum MoleculeType{
  unknown = -1,
  protein = 0,
  ligand = 1,
  dna = 2,
  rna = 3,
  carbohydrate = 4,
  water = 5
} MoleculeType;

typedef enum Geometry{
  POINT = 1,
  LINE = 2,
  TRIANGLE = 3,
  SPHERE = 4,
  CUBE = 6,
  COMPLEX = 7
} Geometry;

struct is_not_neg{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x >= 0);
  }
};
struct is_not_one{
  __host__ __device__
  bool operator()(const int2 x)
  {
    return (x.y == 1);
  }
  __host__ __device__
  bool operator()(const int x)
  {
    return (x == 1);
  }
};


//#define PI 3.1415926535897932384626433832795028841971693993

static inline void ltrim(std::string &s){
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch){
    return !std::isspace(ch);
  }));
}
static inline void rtrim(std::string &s){
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch){
    return !std::isspace(ch);
  }).base(), s.end());
}
static inline void trim(std::string &s){
  ltrim(s);
  rtrim(s);
}

#endif /* COMMON_INCLUDES_H */
