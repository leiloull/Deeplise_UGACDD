#include "AtomTyping.h"

// //ISE SPECIFIC MAPPING -> ["atomName-resName"] = "ISEAtomType"
std::map<std::string, std::string> nameResidueTypingMap;

void fillTypingScheme(std::string pathToAtomTypeCSV){
  std::ifstream data(pathToAtomTypeCSV.c_str());
  std::string line;
  while(std::getline(data,line)){
    std::stringstream lineStream(line);
    std::string atomResName;
    std::string temp;
    std::getline(lineStream,temp,',');
    atomResName = temp;
    std::getline(lineStream,temp,',');
    atomResName += "-" + temp;
    std::getline(lineStream,temp,',');
    nameResidueTypingMap.insert(std::make_pair(atomResName, temp));
    updateAtomTypeMap(temp);
    std::cout<<atomResName<<","<<temp<<std::endl;
  }
}

std::map<std::string, std::string> nameIgnoreListMap;

void fillTypingSchemeIgnoreList(std::string pathToIgnoreListCSV){
  std::ifstream data(pathToIgnoreListCSV.c_str());
  std::string line;
  while(std::getline(data,line)){
    std::stringstream lineStream(line);
    std::string atomResName;
    std::string temp;
    std::getline(lineStream,temp,',');
    atomResName = temp;
    std::getline(lineStream,temp,',');
    atomResName += "-" + temp;
    std::getline(lineStream,temp,',');
    nameIgnoreListMap.insert(std::make_pair(atomResName, temp));
    updateIgnoreListMap(temp);
    std::cout<<atomResName<<","<<temp<<std::endl;
  }
}

//methods using predefined hard coded atom types (will often cause MissingAtomType exception on mutated residues)
//TODO implement this for all types of atoms
void classifyAtomsISE(std::vector<Molecule*> &molecules){
  std::string nameAndResidue;
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        nameAndResidue = (*atom)->name + "-" + (*res)->name;
        if(nameResidueTypingMap[nameAndResidue].length() > 0){
          (*atom)->type = nameResidueTypingMap[nameAndResidue];
          updateAtomTypeMap((*atom)->type);
          (*atom)->sphere.type = atomTypeMap[(*atom)->type];
        }
        else continue;
      }
    }
  }
}

//NOTE only protein atoms other atoms will not be given type
void classifyAtomsISEResidue(std::vector<Molecule*> &molecules){
	std::string nameAndResidue;
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    if((*mol)->type != protein) continue;
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        nameAndResidue = (*atom)->name + "-" + (*res)->name;
        if(nameResidueTypingMap[nameAndResidue].length() > 0){
          (*atom)->type = nameResidueTypingMap[nameAndResidue];
          updateAtomTypeMap((*atom)->type);
          (*atom)->sphere.type = atomTypeMap[(*atom)->type];
        }
        else continue;
      }
    }
  }
}

bool shouldIgnore(std::string nameRes) {
  if (nameIgnoreListMap[nameRes].length() > 0) {
    return true;
  }
  else {
    return false;
  }
}

void exhaustiveClassifyAtomsISEResidue(std::vector<Molecule *> &molecules)
{
  std::string nameAndResidue;
  for (auto mol = molecules.begin(); mol != molecules.end(); ++mol)
  {
    if ((*mol)->type != protein)
      continue;
    for (auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res)
    {
      for (auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom)
      {
        nameAndResidue = (*atom)->name + "-" + (*res)->name;
        (*atom)->type = nameAndResidue;
        updateAtomTypeMap((*atom)->type);
        (*atom)->sphere.type = atomTypeMap[(*atom)->type];
      }
    }
  }
}

//NOTE only nucleic acid other atoms will not be given type
//TODO need to fill in map with these residue atoms
void classifyAtomsISENA(std::vector<Molecule*> &molecules){
  std::string nameAndResidue;
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    if((*mol)->type != dna && (*mol)->type != rna) continue;
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        nameAndResidue = (*atom)->name + "-" + (*res)->name;
        if(nameResidueTypingMap[nameAndResidue].length() > 0){
          (*atom)->type = nameResidueTypingMap[nameAndResidue];
          updateAtomTypeMap((*atom)->type);
          (*atom)->sphere.type = atomTypeMap[(*atom)->type];
        }
        else continue;
      }
    }
  }
}
