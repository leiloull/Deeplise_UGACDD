#ifndef MOLECULE_CUH
#define MOLECULE_CUH
#include "common_includes.h"
#include "bio_maps.h"

struct Sphere;
struct Atom;
struct Residue;
struct Molecule;

extern std::map<std::string, MoleculeType> residueTypeMap;

struct Sphere{
  float3 center;
  float radius;
  int3 molResAtom;
  bool surf;
  unsigned char type;
  MoleculeType mol_type;

  __device__ __host__ Sphere();
	__device__ __host__ Sphere(float3 center);
  __device__ __host__ Sphere(float3 center, float radius);
	__device__ __host__ Sphere(float3 center, float radius, int type);
  __device__ __host__ void printSphere();
};
__device__ __host__ bool compareByX(const Sphere &a, const Sphere &b);
__device__ __host__ bool compareByY(const Sphere &a, const Sphere &b);
__device__ __host__ bool compareByZ(const Sphere &a, const Sphere &b);
__device__ __host__ bool operator<(const Sphere &a, const Sphere &b);
struct SphereX{
  __host__ __device__ bool operator()(const Sphere& a, const Sphere& b){
    return a.center.x < b.center.x;
  }
};
struct SphereY{
  __host__ __device__ bool operator()(const Sphere& a, const Sphere& b){
    return a.center.y < b.center.y;
  }
};
struct SphereZ{
  __host__ __device__ bool operator()(const Sphere& a, const Sphere& b){
    return a.center.z < b.center.z;
  }
};
struct SphereSurf{
  __host__ __device__ bool operator()(const Sphere& a, const Sphere& b){
    return a.surf > b.surf;
  }
};

std::string getElementFromAtomName(std::string &atomName);

struct Atom{

  int id;
	std::string fullDescriptor;//parsed line from whatever file they came from
  std::string element;
  std::string name;
  float betaFactor;
  int charge;
  std::string type;
  float relativeAffinity;

  bool truePositive;//only used for accuracy analysis and only set on protein atoms

  Sphere sphere;
  Residue* parent;

  //may want to use this for multiple occupancy handling, currently removing altLoc B
  //would need to determine max number of altLocs
  float occupancy;
  char altLoc;

	Atom();
  Atom(int id);
  Atom(std::string lineFromFile, FileType fileType);
	virtual ~Atom() = default;

	std::string createPDBDescriptor(int serialNum);
  std::string createPDBDescriptor(int serialNum, float visual);
	void print();

};

struct Residue{

  int id;
  char chain;
  char insertion;
  std::string name;
  float relativeAffinity;

  MoleculeType type;
  Molecule* parent;
  std::vector<Atom*> atoms;

  Residue();
  Residue(Residue* residue);
  Residue(int id);
  Residue(int id, std::vector<Atom*> atoms);
  void setAffinityToMostAffineAtom();
  virtual ~Residue() = default;

};

struct Molecule{

	int id;
  std::string name;
  MoleculeType type;
  std::vector<Residue*> residues;//if size = 0, residue is molecule
  int3 min;
  int3 max;

  Molecule();
  Molecule(int id);
  Molecule(int id, std::vector<Residue*> residues);

  void classify();
  std::vector<Atom*> getAtoms();

  virtual ~Molecule() = default;
};


#endif /* MOLECULE_CUH */
