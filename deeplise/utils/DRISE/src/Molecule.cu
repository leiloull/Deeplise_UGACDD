#include "Molecule.cuh"

std::map<std::string, MoleculeType> residueTypeMap = {
	{"HOH",water},
	{"ATP",ligand},
	{"ADP",ligand},
	{"AMP",ligand},
	{"CDP",ligand},
	{"CTP",ligand},
	{"GMP",ligand},
	{"GDP",ligand},
	{"GTP",ligand},
	{"TMP",ligand},
	{"TTP",ligand},
	{"UMP",ligand},
	{"UDP",ligand},
	{"UTP",ligand},
	{"DA",dna},
	{"DC",dna},
	{"DG",dna},
	{"DT",dna},
	{"DI",dna},
	{"A",rna},
	{"C",rna},
	{"G",rna},
	{"U",rna},
	{"I",rna},
	{"ALA",protein},
	{"CYS",protein},
	{"ASP",protein},
	{"GLU",protein},
	{"PHE",protein},
	{"GLY",protein},
	{"HIS",protein},
	{"ILE",protein},
	{"LYS",protein},
	{"LEU",protein},
	{"MET",protein},
	{"ASN",protein},
	{"PRO",protein},
	{"GLN",protein},
	{"ARG",protein},
	{"SER",protein},
	{"THR",protein},
	{"VAL",protein},
	{"TRP",protein},
	{"TYR",protein},
	{"BMA",carbohydrate},
	{"MAN",carbohydrate},
	{"NAG",carbohydrate},
	{"FUC",carbohydrate},
	{"GLO",carbohydrate},
	{"LBT",carbohydrate},
	{"A2G",carbohydrate},
	{"MAL",carbohydrate},
	{"SUC",carbohydrate},
	{"GAL",carbohydrate}
};


Sphere::Sphere(){
	this->surf = false;
	this->molResAtom = {-1,-1,-1};
	this->center = {0.0f,0.0f,0.0f};
	this->type = 255;
	this->mol_type = unknown;
}
Sphere::Sphere(float3 center){
	this->center = center;
	this->molResAtom = {-1,-1,-1};
	this->surf = false;
	this->type = 255;
	this->mol_type = unknown;
}
Sphere::Sphere(float3 center, float radius){
	this->center = center;
	this->radius = radius;
	this->molResAtom = {-1,-1,-1};
	this->surf = false;
	this->type = 255;
	this->mol_type = unknown;
}
Sphere::Sphere(float3 center, float radius, int type){
	this->center = center;
	this->radius = radius;
	this->type = type;
	this->molResAtom = {-1,-1,-1};
	this->surf = false;
	this->mol_type = unknown;
}
__device__ __host__ void Sphere::printSphere(){
	printf("molResAtom{%d,%d,%d} - %d {%f,%f,%f}\n",this->molResAtom.x,this->molResAtom.y,this->molResAtom.z,this->type,this->center.x,this->center.y,this->center.z);
}

__device__ __host__ bool compareByX(const Sphere &a, const Sphere &b){
  return a.center.x < b.center.x;
}
__device__ __host__ bool compareByY(const Sphere &a, const Sphere &b){
  return a.center.y < b.center.y;
}
__device__ __host__ bool compareByZ(const Sphere &a, const Sphere &b){
  return a.center.z < b.center.z;
}
__device__ __host__ bool operator<(const Sphere &a, const Sphere &b){
  return a.surf > b.surf;
}

Molecule::Molecule(){
	this->id = -1;
	this->name = "-";
	this->type = unknown;
	this->min = {0,0,0};
	this->max = {0,0,0};
}
Molecule::Molecule(int id){
	this->id = id;
}
Molecule::Molecule(int id, std::vector<Residue*> residues){
	this->id = id;
	this->residues = residues;
	this->name = "-";
	this->type = unknown;
	this->classify();
}

void Molecule::classify(){
	std::string name;
	std::string resChecker;
	name = this->name;
	bool nameAnalysisNecessary = false;
	for(auto res = this->residues.begin(); res != this->residues.end(); ++res){
		resChecker = " " + (*res)->name + " ";
		if(residueTypeMap.find((*res)->name) == residueTypeMap.end()){
			(*res)->type = unknown;
		}
		else{
			(*res)->type = residueTypeMap[(*res)->name];
			if(this->type != unknown && (*res)->type != this->type) nameAnalysisNecessary = true;
			else this->type = (*res)->type;
		}
	}
	if(this->type != unknown && !nameAnalysisNecessary) return;
	this->type = unknown;
	std::transform(name.begin(), name.end(), name.begin(), (int (*)(int))tolower);
	if(name.find("protein") != std::string::npos || name.find("ase") != std::string::npos ||
	name.find("domain") != std::string::npos || name.find("chaperone") != std::string::npos ||
	name.find("enzyme") != std::string::npos || name.find("histone") != std::string::npos ||
	name.find("-binding") != std::string::npos || name.find("transporter") != std::string::npos ||
	name.find("regulator") != std::string::npos){
		this->type = protein;
		return;
	}
	else if(name.find("5'-d") != std::string::npos){
		this->type = dna;
		return;
	}
	else if(name.find("5'-r") != std::string::npos){
		this->type = rna;
		return;
	}
	else if(name.find("cofactor") != std::string::npos ||
	name.find("ligand") != std::string::npos ||
	name.find("metabolite") != std::string::npos){
		this->type = ligand;
		return;
	}
}

std::vector<Atom*> Molecule::getAtoms(){
	std::vector<Atom*> atoms;
	for(auto res = this->residues.begin(); res != this->residues.end(); ++res){
		atoms.insert(atoms.end(), (*res)->atoms.begin(), (*res)->atoms.end());
	}
	return atoms;
}

Residue::Residue(){
	this->id = -1;
	this->chain = '-';
	this->insertion = '-';
	this->name = '-';
	this->parent = NULL;
	this->relativeAffinity = 0.0f;
}

Residue::Residue(int id){
	this->id = id;
	this->relativeAffinity = 0.0f;
}

Residue::Residue(int id, std::vector<Atom*> atoms){
	this->id = id;
	this->chain = '-';
	this->insertion = '-';
	this->name = '-';
	this->parent = NULL;
	this->atoms = atoms;
	this->relativeAffinity = 0.0f;
	for(auto atom = this->atoms.begin(); atom != this->atoms.end(); ++atom){
		if((*atom)->relativeAffinity > this->relativeAffinity){
			this->relativeAffinity = (*atom)->relativeAffinity;
		}
	}
}

Residue::Residue(Residue* residue){
	this->id = residue->id;
	this->chain = residue->chain;
	this->insertion = residue->insertion;
	this->name = residue->name;
	this->parent = NULL;
	this->relativeAffinity = 0.0f;
	if(residue->atoms.size() != 0){
		this->atoms = residue->atoms;
		for(auto atom = this->atoms.begin(); atom != this->atoms.end(); ++atom){
			if((*atom)->relativeAffinity > this->relativeAffinity){
				this->relativeAffinity = (*atom)->relativeAffinity;
			}
		}
	}
}

void Residue::setAffinityToMostAffineAtom(){
	for(auto atom = this->atoms.begin(); atom != this->atoms.end(); ++atom){
		if((*atom)->relativeAffinity > this->relativeAffinity){
			this->relativeAffinity = (*atom)->relativeAffinity;
		}
	}
}

Atom::Atom() {
	this->id = -1;
	this->element = "-";
	this->name = "-";
	this->betaFactor = 0.0f;
	this->type = "---";
	this->relativeAffinity = 0.0f;
	this->parent = NULL;
	this->truePositive = false;

}

Atom::Atom(int id){
	this->id = id;
	this->element = "-";
	this->name = "-";
	this->betaFactor = 0.0f;
	this->type = "---";
	this->relativeAffinity = 0.0f;
	this->parent = NULL;
	this->truePositive = false;
}

Atom::Atom(std::string lineFromFile, FileType fileType){
	if(fileType == PDB){
		this->truePositive = false;
		this->relativeAffinity = 0.0f;
		this->fullDescriptor = lineFromFile;
		this->id = stoi(lineFromFile.substr(6, 5));
		this->name = lineFromFile.substr(12, 4);

		//currently not using this
		this->altLoc = lineFromFile.at(16);
		this->occupancy = stof(lineFromFile.substr(54, 6));
		this->parent = NULL;
		this->type = "---";

		float3 center = {
			stof(lineFromFile.substr(30, 8)),
			stof(lineFromFile.substr(38, 8)),
			stof(lineFromFile.substr(46, 8))
		};

		this->betaFactor = stof(lineFromFile.substr(60, 6));

		this->element = (lineFromFile.length() >= 79) ? lineFromFile.substr(76, 2) : throw PDBFormatException("no element");
		std::string temp = (lineFromFile.length() >= 81) ? lineFromFile.substr(78, 2) : "";
		trim(temp);
		if(temp.length() != 0){
			this->charge = stoi(temp);
		}
		else{
			this->charge = 0;
		}
		trim(this->name);
		trim(this->element);
		this->sphere = Sphere(center,radius[this->element]);
		if(this->name.length() == 0){
			this->name = this->element;
		}
	}
	else{
		throw ISEException_runtime("unsupported filetype");
	}
}

std::string Atom::createPDBDescriptor(int serialNum){
	if(this->fullDescriptor.length() == 0){
		this->fullDescriptor = "ATOM  ";
		std::string temp = std::to_string(serialNum);
		while(temp.length() < 5) temp = " " + temp;
		this->fullDescriptor += temp + " ";
		temp = this->name;
		while(temp.length() < 3) temp = " " + temp;
		this->fullDescriptor += temp + " ";

		if(this->parent != NULL){
			temp = this->parent->name;
			while(temp.length() < 3) temp = " " + temp;
			this->fullDescriptor += temp + " A";
			temp = std::to_string(this->parent->id);
			while(temp.length() < 4) temp = " " + temp;
			this->fullDescriptor += temp + "1   ";
		}
		else{
			std::cout<<"ERROR cannot write pdb atom string without parent residue set"<<std::endl;
			exit(-1);
		}

		char sz[64];

		sprintf(sz, "%.3lf", this->sphere.center.x);
		temp = std::string(sz);
		while(temp.length() < 8) temp = " " + temp;
		this->fullDescriptor += temp + " ";

		sprintf(sz, "%.3lf", this->sphere.center.y);
		temp = std::string(sz);
		while(temp.length() < 8) temp = " " + temp;
		this->fullDescriptor += temp + " ";

		sprintf(sz, "%.3lf", this->sphere.center.z);
		temp = std::string(sz);
		while(temp.length() < 8) temp = " " + temp;
		this->fullDescriptor += temp + " ";

		temp = "1.23";
		while(temp.length() < 6) temp = " " + temp;
		this->fullDescriptor += temp + " ";

		sprintf(sz, "%.2lf", this->betaFactor);
		temp = std::string(sz);
		while(temp.length() < 6) temp = " " + temp;
		this->fullDescriptor += temp + " ";

		this->fullDescriptor += (this->element.length() < 2) ? " " + this->element : this->element;
		
		if(this->charge > 0){
			this->fullDescriptor += "+" + std::to_string(this->charge);
		}
		else if(this->charge < 0){
			this->fullDescriptor += std::to_string(this->charge);
		}
	}
	return this->fullDescriptor;
}
std::string Atom::createPDBDescriptor(int serialNum, float visual){
	std::ostringstream out;
	out.precision(2);
	out << std::fixed << visual;
	std::string visual_string = out.str();
	while(visual_string.length() < 6) visual_string = " " + visual_string;
	std::string to_return;
	this->createPDBDescriptor(serialNum);

	to_return = this->fullDescriptor.substr(0,60) + visual_string + this->fullDescriptor.substr(67);

	return to_return;
}
void Atom::print() {
	if(this->fullDescriptor.length() == 0){
		this->createPDBDescriptor(this->id);
	}
	std::cout<<this->fullDescriptor<<std::endl;
}
