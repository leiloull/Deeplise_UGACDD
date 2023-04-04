#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include "Molecule.cuh"
#include "Triangle.cuh"
#include "bio_maps.h"
#include "cuda_util.cuh"
#include "json.hpp"//TODO find other library in which this is not necessary

bool fileExists(std::string fileName);
bool directoryExists(std::string dirPath);
void parseDirectory(std::string dirPath, std::vector<std::string> &files);
void parseIdentifierTXT(std::string txt, std::vector<std::string> &ids);
void parseArgs(std::string pathToTypingScheme, std::set<MoleculeType> &bindConfig, int &inputType, std::vector<std::string> &input,
  int &runConfig, ConnectorType &storageMethod, const int &numArgs, char** args);

std::string getStringFromJSON(std::string pathToFile);
std::string createSevenCharInt(int i);
std::string createFourCharInt(int i);

std::vector<Molecule*> separateMolecules(const std::string &metadata, std::vector<Residue*> residues, FileType fileType);
void parsePDB(std::string pathToFile, std::vector<Molecule*> &molecules, std::string &metadata, std::string &identifier, std::string &extraInfo);
void parsePDB(std::string pathToFile, std::vector<Residue*> &residues);

std::string createMolecularJSON(const std::vector<Molecule*> &molecules, const std::string &identifier);
void readMolecularJSON(std::string particleStr, std::vector<Molecule*> &molecules, std::string &identifier);
std::string createTrianglesJSON(const std::vector<Triangle> &triangles);
void readTrianglesJSON(std::string triangleStr, std::vector<Triangle> &triangles);

void prepareVisualizationFile(const std::vector<Molecule*> &molecules, std::string identifier, bool residueScoring);
void prepareVisualizationFile(const std::vector<Molecule*> &molecules, std::string identifier, MoleculeType type, bool residueScoring);
void prepareVisualizationFile(const Molecule* molecule, std::string identifier, bool residueScoring);
void writeAtomTypeChecker(const std::vector<Molecule*> &molecules, const std::string &identifier);
void writeAtomTypeChecker(const Molecule* molecule, const std::string &identifier);
void writeSurfaceChecker(const std::vector<Molecule*> &molecules, const std::string &identifier);
void writeSurfaceChecker(const Molecule* molecule, std::string identifier);
void writeTrainingLog(std::string identifier, std::vector<Triangle> triangles, std::vector<std::string> datavector);

#endif /* IO_UTIL_H */
