#ifndef PARTICLELIST_CUH
#define PARTICLELIST_CUH

#include "common_includes.h"
#include "bio_maps.h"
#include "Molecule.cuh"
#include "io_util.h"
#include "Octree.cuh"
#include "AtomTyping.h"
#include "Unity.cuh"

class ParticleList {

private:

	void findMinMax();
	void removeHydrogens();
	void removeWater();
	void calibrateSpheres();

public:

	FileType fileType;

	std::set<MoleculeType> typesInOctree;
	Octree* octree;

	float3 min;
	float3 max;

	std::vector<Molecule*> molecules;

	std::string identifier;
	std::string metadata;
	std::string filePath;
	std::string extraInfo;

	ParticleList();
	ParticleList(std::string filePath);
	ParticleList(std::vector<Molecule*> molecules, std::string identifier);


	ParticleList(std::vector<Atom*> atoms, std::string identifier);
	ParticleList(std::vector<Residue*> residues, std::string identifier);

	~ParticleList();

	//allows for outside methods to take over atom typing
	void classifyAtoms(void (*atomTypeProtocol)(std::vector<Molecule*>&));
	void classifyAtoms(void (*atomTypeProtocol)(std::vector<Molecule*>&,Octree*));
	void classifyAtoms(void (*atomTypeProtocol)(std::vector<Molecule*>&,std::string));
	void convertAtomTypes(std::map<std::string,std::string> &converter);
	bool checkForTypingContinuity(MoleculeType type);
	bool checkForTypingContinuity(std::set<MoleculeType> types);

	void determineResidueScores();

	void printAll();

	std::vector<Atom*> getAtoms();
	std::vector<Atom*> getAtoms(MoleculeType type);
	std::vector<Atom*> getAtoms(std::set<MoleculeType> types);
	std::vector<Residue*> getResidues();
	std::vector<Residue*> getResidues(MoleculeType type);
	std::vector<Residue*> getResidues(std::set<MoleculeType> types);
	std::vector<Molecule*> getMolecules(MoleculeType type);
	std::vector<Molecule*> getMolecules(std::set<MoleculeType> types);

	//TODO evaluate need for this
	jax::Unity<Sphere>* getSpheres(bool keepHydrogens, bool onlySpatial);
	jax::Unity<Sphere>* getSpheres(bool keepHydrogens, bool onlySpatial, MoleculeType moleculeType);
	jax::Unity<Sphere>* getSpheres(bool keepHydrogens, bool onlySpatial, std::set<MoleculeType> moleculeTypes);

	Octree* createOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface);
	Octree* createOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, MoleculeType moleculeType);
	Octree* createOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, std::set<MoleculeType> moleculeTypes);
	void setOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface);
	void setOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, MoleculeType moleculeType);
	void setOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, std::set<MoleculeType> moleculeTypes);


};

#endif /* PARTICLELIST_CUH */
