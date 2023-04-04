#ifndef ATOMTYPING_H
#define ATOMTYPING_H

#include "common_includes.h"
#include "bio_maps.h"
#include "Molecule.cuh"
#include "Octree.cuh"
#include "io_util.h"
#include "cuda_util.cuh"

/*
THIS FILE IS FOR PLACING CUSTOM ATOM TYPE DEFINTIONS
*/

//FILLED IN ATOMTYPING_CU
//ISE SPECIFIC MAPPING -> ["atomName-resName"] = "ISEAtomType"
//TODO ADD DNA AND RNA TO MAP
void fillTypingScheme(std::string pathToAtomTypeCSV);
void fillTypingSchemeIgnoreList(std::string pathToIgnoreListCSV);
extern std::map<std::string, std::string> nameResidueTypingMap;

void classifyAtomsISE(std::vector<Molecule*> &molecules);
void classifyAtomsISEResidue(std::vector<Molecule*> &molecules);
void exhaustiveClassifyAtomsISEResidue(std::vector<Molecule *> &molecules);
void classifyAtomsISENA(std::vector<Molecule*> &molecules);
bool shouldIgnore(std::string nameRes);

#endif /* ATOMTYPING_H */
