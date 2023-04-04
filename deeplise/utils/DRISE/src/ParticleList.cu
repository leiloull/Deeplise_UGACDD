#include "ParticleList.cuh"

ParticleList::ParticleList(){
	this->octree = NULL;
	this->min = {FLT_MAX,FLT_MAX,FLT_MAX};
	this->max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
	this->identifier = "---";
	this->fileType = NO_FILE;
}
ParticleList::ParticleList(std::string filePath) {
	if(filePath.find(".pdb") != std::string::npos){
		this->fileType = PDB;
		std::cout<<"Using pdb reader"<<std::endl;
		this->identifier = filePath.substr(filePath.length() - 9, 4);
		parsePDB(filePath, this->molecules, this->metadata, this->identifier, this->extraInfo);

	}
	else if(filePath.find(".json") != std::string::npos){
		this->fileType = JSON;
		std::cout<<"Using json reader"<<std::endl;
		readMolecularJSON(getStringFromJSON(filePath), this->molecules, this->identifier);
	}
	else{
		std::cout<<filePath<<" is not a valid path to retrieve particle info"<<std::endl;
	}
	this->filePath = filePath;

	this->findMinMax();

	this->octree = NULL;
	this->removeHydrogens();
	this->removeWater();
	this->calibrateSpheres();
}
ParticleList::ParticleList(std::vector<Molecule*> molecules, std::string identifier){
	this->molecules = molecules;
	this->identifier = identifier;
	this->filePath = "n/a";
	this->findMinMax();
	this->octree = NULL;
	this->removeHydrogens();
	this->removeWater();
	this->calibrateSpheres();
	this->fileType = NO_FILE;
}
ParticleList::ParticleList(std::vector<Residue*> residues, std::string identifier){
	this->identifier = identifier;
	int mol_id = 0;
	for(auto res = residues.begin(); res != residues.end(); ++res){
		if((*res)->parent == NULL){
			std::vector<Residue*> temp;
			temp.push_back((*res));
			(*res)->parent = new Molecule(mol_id++, temp);
		}
		else if((*res)->parent->residues.size() == 0){
			(*res)->parent->residues.push_back((*res));
			(*res)->parent->id = mol_id++;
		}
		this->molecules.push_back((*res)->parent);
	}
	this->removeHydrogens();
	this->removeWater();
	this->calibrateSpheres();
	this->fileType = NO_FILE;
}
ParticleList::ParticleList(std::vector<Atom*> atoms, std::string identifier){
	this->identifier = identifier;
	int mol_id = 0;
	int res_id = 0;
	for(auto atom = atoms.begin(); atom != atoms.end(); ++atom){
		if((*atom)->parent == NULL){
			std::vector<Atom*> temp;
			temp.push_back((*atom));
			(*atom)->parent = new Residue(res_id++, temp);
			std::vector<Residue*> temp2;
			temp2.push_back((*atom)->parent);
			(*atom)->parent->parent = new Molecule(mol_id++, temp2);
		}
		else if((*atom)->parent->atoms.size() == 0){
			(*atom)->parent->atoms.push_back((*atom));
			(*atom)->parent->id = res_id++;
			if((*atom)->parent->parent == NULL){
				std::vector<Residue*> temp;
				temp.push_back((*atom)->parent);
				(*atom)->parent->parent = new Molecule(mol_id++, temp);
			}
			else if((*atom)->parent->parent->residues.size() == 0){
				(*atom)->parent->parent->residues.push_back((*atom)->parent);
				(*atom)->parent->parent->id = mol_id++;
			}
		}
		this->molecules.push_back((*atom)->parent->parent);
	}
	this->removeHydrogens();
	this->removeWater();
	this->calibrateSpheres();
	this->fileType = NO_FILE;
}

ParticleList::~ParticleList(){
	if(this->octree != NULL){
		delete this->octree;
	}
	for(int m = 0; m < this->molecules.size(); ++m){
		for(int r = 0; r < this->molecules[m]->residues.size(); ++r){
			for(int a = 0; a < this->molecules[m]->residues[r]->atoms.size(); ++a){
				delete this->molecules[m]->residues[r]->atoms[a];
			}
			delete this->molecules[m]->residues[r];
		}
		delete this->molecules[m];
	}
}

void ParticleList::classifyAtoms(void (*atomTypeProtocol)(std::vector<Molecule*>&)){
	atomTypeProtocol(this->molecules);
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((*atom)->type != "---") updateAtomTypeMap((*atom)->type);
			}
		}
	}
}
void ParticleList::classifyAtoms(void (*atomTypeProtocol)(std::vector<Molecule*>&,Octree*)){
	atomTypeProtocol(this->molecules, this->octree);
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((*atom)->type != "---") updateAtomTypeMap((*atom)->type);
			}
		}
	}
}
void ParticleList::classifyAtoms(void (*atomTypeProtocol)(std::vector<Molecule*>&,std::string)){
	atomTypeProtocol(this->molecules, this->filePath);
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((*atom)->type != "---") updateAtomTypeMap((*atom)->type);
			}
		}
	}
}

void ParticleList::convertAtomTypes(std::map<std::string,std::string> &converter){
	atomTypeMap.clear();
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if(((*atom)->type).length() > 0 && converter[((*atom)->type)].length() > 0) {
					(*atom)->type = converter[((*atom)->type)];
					updateAtomTypeMap((*atom)->type);
				}
				else{
					writeAtomTypeChecker(this->molecules, this->identifier);
					throw MissingAtomType("map does not convert all atoms");
				}
			}
		}
	}
}
bool ParticleList::checkForTypingContinuity(MoleculeType type){
	bool to_return = false;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		if((*mol)->type != type) continue;
		to_return = true;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			 std::vector<int> deleteVector;
			 int numDeleted = 0;
			 int atomIndex = 0;
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((*atom)->type == "---"){
					 if (shouldIgnore((*atom)->name + "-" + (*res)->name)) {
						deleteVector.push_back(atomIndex);
					 }
					 else {
					 	throw MissingAtomType((*atom)->createPDBDescriptor((*atom)->id));
					 }
				}
				else if ((*atom)->type == "IGR") {

					deleteVector.push_back(atomIndex);

				}
				atomIndex += 1;
			}
			 for (int atomIndexToDelete : deleteVector) {

			 	(*res)->atoms.erase((*res)->atoms.begin() + atomIndexToDelete - numDeleted);
			 	numDeleted += 1;

			 }
		}
	}
	return to_return;
}
// bool ParticleList::checkForTypingContinuity(std::set<MoleculeType> types){
// 	bool found = false;
// 	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
// 		found = false;
// 		for(auto type = types.begin(); type != types.end(); ++type){
// 			if((*mol)->type == (*type)){
// 				found = true;
// 				break;
// 			}
// 		}
// 		if(!found) continue;
// 		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
// 			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
// 				if((*atom)->type == "---"){
// 					return false;
// 				}
// 			}
// 		}
// 	}
// 	return true;
// }

void ParticleList::determineResidueScores(){
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      (*res)->setAffinityToMostAffineAtom();
    }
  }
}

void ParticleList::printAll(){
	std::cout<< this->identifier <<std::endl;
	std::cout<< this->metadata <<std::endl;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				(*atom)->print();
			}
		}
	}
	std::cout<< this->extraInfo <<std::endl;
}

void ParticleList::removeHydrogens(){
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((*atom)->element == "H"){
					(*res)->atoms.erase(atom);
					atom--;
				}
			}
		}
	}
}
void ParticleList::removeWater(){
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			if((*res)->name == "HOH"){
				(*mol)->residues.erase(res);
				res--;
			}
		}
	}
}

void ParticleList::findMinMax(){
	this->min.x = FLT_MAX;
	this->min.y = FLT_MAX;
	this->min.z = FLT_MAX;
	this->max.x = -FLT_MAX;
	this->max.y = -FLT_MAX;
	this->max.z = -FLT_MAX;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if(this->min.x > (*atom)->sphere.center.x) this->min.x = (*atom)->sphere.center.x;
				if(this->min.y > (*atom)->sphere.center.y) this->min.y = (*atom)->sphere.center.y;
				if(this->min.z > (*atom)->sphere.center.z) this->min.z = (*atom)->sphere.center.z;
				if(this->max.x < (*atom)->sphere.center.x) this->max.x = (*atom)->sphere.center.x;
				if(this->max.y < (*atom)->sphere.center.y) this->max.y = (*atom)->sphere.center.y;
				if(this->max.z < (*atom)->sphere.center.z) this->max.z = (*atom)->sphere.center.z;
			}
		}
	}

	this->max.x += 2;
	this->max.y += 2;
	this->max.z += 2;
	this->min.x -= 2;
	this->min.y -= 2;
	this->min.z -= 2;
}

void ParticleList::calibrateSpheres(){
	int3 molResAtom = {0,0,0};
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		(*mol)->classify();
		molResAtom.y = 0;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			molResAtom.z = 0;
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				(*atom)->sphere.molResAtom = molResAtom;
				(*atom)->sphere.mol_type = (*mol)->type;
				(*atom)->sphere.type = ((*atom)->type != "---") ? atomTypeMap[(*atom)->type] : 255;
				molResAtom.z++;
			}
			molResAtom.y++;
		}
		molResAtom.x++;
	}
}

std::vector<Atom*> ParticleList::getAtoms(){
	std::vector<Atom*> atoms;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			atoms.insert(atoms.end(), (*res)->atoms.begin(), (*res)->atoms.end());
		}
	}
	return atoms;
}
std::vector<Atom*> ParticleList::getAtoms(MoleculeType type){
	std::vector<Atom*> atoms;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		if((*mol)->type != type) continue;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			atoms.insert(atoms.end(), (*res)->atoms.begin(), (*res)->atoms.end());
		}
	}
	return atoms;
}
std::vector<Atom*> ParticleList::getAtoms(std::set<MoleculeType> types){
	std::vector<Atom*> atoms;
	bool found = false;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		found = false;
		for(auto type = types.begin(); type != types.end(); ++type){
			if((*type) == (*mol)->type){
				found = true;
				break;
			}
		}
		if(!found) continue;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			atoms.insert(atoms.end(), (*res)->atoms.begin(), (*res)->atoms.end());
		}
	}
	return atoms;
}
std::vector<Residue*> ParticleList::getResidues(){
	std::vector<Residue*> residues;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		residues.insert(residues.end(), (*mol)->residues.begin(), (*mol)->residues.end());
	}
	return residues;
}
std::vector<Residue*> ParticleList::getResidues(MoleculeType type){
	std::vector<Residue*> residues;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		if((*mol)->type != type) continue;
		residues.insert(residues.end(), (*mol)->residues.begin(), (*mol)->residues.end());
	}
	return residues;
}
std::vector<Residue*> ParticleList::getResidues(std::set<MoleculeType> types){
	std::vector<Residue*> residues;
	bool found = false;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		found = false;
		for(auto type = types.begin(); type != types.end(); ++type){
			if((*type) == (*mol)->type){
				found = true;
				break;
			}
		}
		if(found) residues.insert(residues.end(), (*mol)->residues.begin(), (*mol)->residues.end());
	}
	return residues;
}
std::vector<Molecule*> ParticleList::getMolecules(MoleculeType type){
	std::vector<Molecule*> molecules;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		if(type == (*mol)->type){
			molecules.push_back((*mol));
		}
	}
	return molecules;
}
std::vector<Molecule*> ParticleList::getMolecules(std::set<MoleculeType> types){
	std::vector<Molecule*> molecules;
	bool found = false;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		found = false;
		for(auto type = types.begin(); type != types.end(); ++type){
			if((*type) == (*mol)->type){
				found = true;
				break;
			}
		}

		if(found) molecules.push_back((*mol));
	}
	return molecules;
}

Unity<Sphere>* ParticleList::getSpheres(bool keepHydrogens, bool onlySurface){
	int numSpheres = 0;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			if(keepHydrogens && !onlySurface) numSpheres += (int) (*res)->atoms.size();
			else{
				for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
					if(((*atom)->element != "H" || keepHydrogens) && (!onlySurface || (*atom)->sphere.surf)) ++numSpheres;
				}
			}
		}
	}
	Sphere* spheres = new Sphere[numSpheres];
	int i = 0;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((!keepHydrogens && (*atom)->element == "H") || (onlySurface && !(*atom)->sphere.surf)) continue;
				spheres[i++] = (*atom)->sphere;
			}
		}
	}
	return new Unity<Sphere>(spheres,numSpheres, cpu);
}
Unity<Sphere>* ParticleList::getSpheres(bool keepHydrogens, bool onlySurface, MoleculeType moleculeType){
	int numSpheres = 0;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		if((*mol)->type != moleculeType) continue;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			if(keepHydrogens && !onlySurface) numSpheres += (int) (*res)->atoms.size();
			else{
				for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
					if(((*atom)->element != "H" || keepHydrogens) && (!onlySurface || (*atom)->sphere.surf)) ++numSpheres;
				}
			}
		}
	}
	if(numSpheres == 0){
		throw LackOfTypeException("cannot find types");
	}
	Sphere* spheres = new Sphere[numSpheres];
	int i = 0;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		if((*mol)->type != moleculeType) continue;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((!keepHydrogens && (*atom)->element == "H") || (onlySurface && !(*atom)->sphere.surf)) continue;
				spheres[i++] = (*atom)->sphere;
			}
		}
	}
	return new Unity<Sphere>(spheres, numSpheres, cpu);
}
Unity<Sphere>* ParticleList::getSpheres(bool keepHydrogens, bool onlySurface, std::set<MoleculeType> moleculeTypes){
	int numSpheres = 0;
	bool found = false;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		found = false;
		for(auto type = moleculeTypes.begin(); type != moleculeTypes.end(); ++type){
			if((*mol)->type == (*type)){
				found = true;
				break;
			}
		}
		if(!found) continue;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){

			if(keepHydrogens && !onlySurface) numSpheres += (int) (*res)->atoms.size();
			else{
				for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
					if(((*atom)->element != "H" || keepHydrogens) && (!onlySurface || (*atom)->sphere.surf)) ++numSpheres;
				}
			}
		}

	}

	if(numSpheres == 0){
		throw LackOfTypeException("cannot find types");
	}

	Sphere* spheres = new Sphere[numSpheres];
	int i = 0;
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		found = false;
		for(auto type = moleculeTypes.begin(); type != moleculeTypes.end(); ++type){
			if((*mol)->type == (*type)){
				found = true;
				break;
			}
		}
		if(!found) continue;
		for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
			for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
				if((!keepHydrogens && (*atom)->element == "H") || (onlySurface && !(*atom)->sphere.surf)) continue;
				spheres[i++] = (*atom)->sphere;
			}
		}
	}
	return new Unity<Sphere>(spheres, numSpheres, cpu);
}

Octree* ParticleList::createOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface){
	return new Octree(this->getSpheres(keepHydrogens, onlySurface), 2.0f, createVEFArrays);
}
Octree* ParticleList::createOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, MoleculeType moleculeType){
	return new Octree(this->getSpheres(keepHydrogens, onlySurface, moleculeType), 2.0f, createVEFArrays);
}
Octree* ParticleList::createOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, std::set<MoleculeType> moleculeTypes){
	return new Octree(this->getSpheres(keepHydrogens, onlySurface, moleculeTypes), 2.0f, createVEFArrays);
}

void ParticleList::setOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface){
	if(this->octree != NULL){
		delete this->octree;
		this->typesInOctree.clear();
	}
	this->octree = new Octree(this->getSpheres(keepHydrogens, onlySurface), 2.0f, createVEFArrays);
	for(auto mol = this->molecules.begin(); mol != this->molecules.end(); ++mol){
		this->typesInOctree.insert((*mol)->type);
	}
}
void ParticleList::setOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, MoleculeType moleculeType){
	bool alreadySet = false;
	for(auto currentType = this->typesInOctree.begin(); currentType != this->typesInOctree.end(); ++currentType){
		alreadySet = true;
		if((*currentType) != moleculeType){
			alreadySet = false;
			break;
		}
	}
	if(alreadySet){
		std::cout<<"WARNING octree already set with this configuration"<<std::endl;
		return;
	}
	if(this->octree != NULL){
		delete this->octree;
		this->typesInOctree.clear();
	}
	this->octree = new Octree(this->getSpheres(keepHydrogens, onlySurface, moleculeType), 2.0f, createVEFArrays);
	this->typesInOctree.insert(moleculeType);
}
void ParticleList::setOctree(bool createVEFArrays, bool keepHydrogens, bool onlySurface, std::set<MoleculeType> moleculeTypes){
	bool alreadySet = false;
	for(auto currentType = this->typesInOctree.begin(); currentType != this->typesInOctree.end(); ++currentType){
		alreadySet = false;
		for(auto newType = moleculeTypes.begin(); newType != moleculeTypes.end(); ++newType){
			if((*newType) == (*currentType)){
				alreadySet = true;
				break;
			}
		}
		if(!alreadySet) break;
	}
	if(alreadySet){
		std::cout<<"WARNING octree already set with this configuration"<<std::endl;
		return;
	}
	if(this->octree != NULL){
		delete this->octree;
		this->typesInOctree.clear();
	}
	this->octree = new Octree(this->getSpheres(keepHydrogens, onlySurface, moleculeTypes), 2.0f, createVEFArrays);
	for(auto type = moleculeTypes.begin(); type != moleculeTypes.end(); ++type){
		this->typesInOctree.insert((*type));
	}
}
