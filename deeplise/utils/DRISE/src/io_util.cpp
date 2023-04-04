#include "io_util.h"

bool fileExists(std::string fileName){
    std::ifstream infile(fileName);
    return infile.good();
}
bool directoryExists(std::string dirPath){
    if(dirPath.c_str() == NULL) return false;
    DIR *dir;
    bool bExists = false;
    dir = opendir(dirPath.c_str());
    if(dir != NULL){
      bExists = true;
      (void) closedir(dir);
    }
    return bExists;
}
void parseDirectory(std::string dirPath, std::vector<std::string> &files){
  DIR* dir;
  if (NULL == (dir = opendir(dirPath.c_str()))){
    printf("Error : Failed to open input directory %s\n",dirPath.c_str());
    exit(-1);
  }
  struct dirent* in_file;
  while((in_file = readdir(dir)) != NULL){
    std::string currentFileName = in_file->d_name;
    if(currentFileName.find(".pdb") == std::string::npos &&
    currentFileName.find(".json") == std::string::npos){
      std::cout<<currentFileName<<" is directory or incompatible file type."<<std::endl;
      continue;
    }
    currentFileName = dirPath + "/" + currentFileName;
    files.push_back(currentFileName);
  }
  closedir(dir);
}
void parseIdentifierTXT(std::string txt, std::vector<std::string> &ids){
  std::ifstream in;
  std::string line = "";
  std::string value = "";
  in.open(txt);
  if(in.is_open()){
    while(getline(in, line)){
      std::stringstream buffer(line);
      while(getline(buffer, value, ',')){
        trim(value);
        if(value.length() != 4){
          std::cout<<"pdb id txt list format is invalid (ids separated by commas)"<<std::endl;
          in.close();
          exit(-1);
        }
        ids.push_back(value);
      }
    }
    in.close();
  }
}
void parseArgs(std::string pathToTypingScheme, std::set<MoleculeType> &bindConfig, int &inputType, std::vector<std::string> &input,
  int &runConfig, ConnectorType &storageMethod, const int &numArgs, char** args){

    std::map<std::string, int> parameters;
    parameters["-i"] = 0;//input (only can have 1)
    parameters["-c"] = 1;//storageConnector config (only can have 1)
    parameters["-p"] = 2;//process mode (only can have 1)
    parameters["-m"] = 3;//moleculeType (can have more than one of these)
    parameters["-h"] = 4;//help
    parameters["-help"] = 5;
    parameters["-a"] = 6;//path to typing scheme for atom types
    if(numArgs < 2){
      std::cout << "for help ./exe -h"<<std::endl;
      exit(-1);
    }
    else if (numArgs >= 2){
      std::string temp = "";
      std::ifstream readme;
      std::string line = "";
      std::cout<<"--------------------USER INPUT---------------------"<<std::endl;
      for(int i = 1; i < numArgs; ++i) {
        switch(parameters[args[i]]){
          case 0:
            if(numArgs == ++i){
              std::cout<<"Error empty field: input"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            temp = args[i];
            std::cout<<"Input = ";
            if((temp.find(".pdb") != std::string::npos || temp.find(".json") != std::string::npos) && fileExists(temp)){
              input.push_back(temp);
              inputType = 0;
              std::cout<<temp<<std::endl;
            }
            else if(directoryExists(temp)){
              parseDirectory(temp, input);
              inputType = 0;
              std::cout<<"directory with "<<input.size()<<" files:"<<std::endl;
              for(auto it = input.begin(); it != input.end(); ++it){
                std::cout<<"\t-"<<(*it)<<std::endl;
              }
            }
            else if(temp.find(".txt") != std::string::npos && fileExists(temp)){
              parseIdentifierTXT(temp, input);
              inputType = 1;
              std::cout<<"text file with list of identifier's:"<<std::endl;
              for(int i = 0; i < (int)input.size(); ++i){
                std::cout<<"\t-"<<input[i]<<std::endl;
              }
            }
            else if(temp.length() == 4){
              input.push_back(temp);
              inputType = 1;
              std::cout<<temp<<std::endl;
            }
            else{
              std::cout<<"Cannot use "<<temp<<" as input"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            break;
          case 1:
            if(numArgs == ++i){
              std::cout<<"Error empty field: storage method"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            temp = args[i];
            std::cout<<"Storage Method = ";
            if(temp =="l" || temp == "local"){
              storageMethod = local_database;
              std::cout<<"local database"<<std::endl;
            }
            else if(temp == "r" || temp == "remote"){
              storageMethod = remote_database;
              std::cout<<"remote database"<<std::endl;
            }
            else if(temp == "f" || temp == "file" || temp == "localFile" || temp == "localfile"){
              storageMethod = file_io;
              std::cout<<"local file io"<<std::endl;
            }
            else{
              std::cout<<"Please only use storageConnector configurations listed in documentation"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            break;
          case 2:
            if(numArgs == ++i){
              std::cout<<"Error empty field: run configuration"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            temp = args[i];
            std::cout<<"Run Configuration = ";
            if(temp == "a" || temp == "at" || temp == "atomtype"){
              runConfig = 0;
              std::cout<<"atom type classification"<<std::endl;
            }
            else if(temp == "s" || temp == "surf" || temp == "surface"){
              runConfig = 1;
              std::cout<<"atom type and surface classification"<<std::endl;
            }
            else if(temp == "t" || temp == "tri" || temp == "train"){
              runConfig = 2;
              std::cout<<"triangle training (atom type, surface, graph building)"<<std::endl;
            }
            else if(temp == "bsp" || temp == "predict" || temp =="bindingSite" || temp =="bindingsite"
            || temp == "BindingSite" || temp == "process" || temp == "Process"){
              runConfig = 3;
              std::cout<<"binding site prediction"<<std::endl;
            }
            else{
              std::cout<<"Please only use processing configurations listed in documentation"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            break;
          case 3:
            if(numArgs == ++i){
              std::cout<<"Error empty field: binding type"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            temp = args[i];
            std::cout<<"Molecule types permitted in computation:"<<std::endl;
            if(temp == "p" || temp == "prot" || temp =="protein" || temp == "Protein"){
              bindConfig.insert(protein);
              std::cout<<"\t-Proteins"<<std::endl;
            }
            else if(temp == "l" || temp == "lig" || temp == "ligand" || temp == "Ligand"){
              bindConfig.insert(ligand);
              std::cout<<"\t-Ligands"<<std::endl;
            }
            else if(temp == "d" || temp =="dna" || temp == "DNA"){
              bindConfig.insert(dna);
              std::cout<<"\t-DNA"<<std::endl;
            }
            else if(temp == "r" || temp == "rna" || temp =="RNA"){
              bindConfig.insert(rna);
              std::cout<<"\t-RNA"<<std::endl;
            }
            else if(temp == "c" || temp == "carb" || temp == "carbohydrate" || temp == "Carbohydrate"){
              bindConfig.insert(carbohydrate);
              std::cout<<"-Carbohydrates"<<std::endl;
            }
            else{
              std::cout<<"Please only use molecule configurations listed in documentation"<<std::endl;
              std::cout << "for help ./exe -h"<<std::endl;
              exit(-1);
            }
            break;
          case 4:
          case 5:
            readme.open("README.txt");
            if(readme.is_open()){
              while(getline(readme, line)){
                std::cout << line << std::endl;
              }
              readme.close();
            }
            exit(0);
          case 6:
            pathToTypingScheme = args[++i];
            break;
          default:
            exit(-1);
        }
      }
    }
}

std::string getStringFromJSON(std::string pathToFile){
  std::ifstream infile(pathToFile);
  if(infile.is_open()){
    std::string file_contents{std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>()};
    infile.close();
    return file_contents;
  }
  else{
    std::cout<<"ERROR opening "+pathToFile<<std::endl;
    exit(-1);
  }
}
std::string createSevenCharInt(int i) {
  std::string strInt;
  if (i < 10) {
    strInt = "      " + std::to_string(i);
  }
  else if (i < 100) {
    strInt = "     " + std::to_string(i);
  }
  else if (i < 1000) {
    strInt = "    " + std::to_string(i);
  }
  else if (i < 10000) {
    strInt = "   " + std::to_string(i);
  }
  else{
    strInt = "  " + std::to_string(i);
  }
  return strInt;
}
std::string createFourCharInt(int i) {
  std::string strInt;
  if (i < 10) {
    strInt = "   " + std::to_string(i);
  } else if (i < 100) {
    strInt = "  " + std::to_string(i);
  } else if (i < 1000) {
    strInt = " " + std::to_string(i);
  } else {
    strInt = std::to_string(i);
  }
  return strInt;
}

//for PDBs
std::vector<Molecule*> separateMolecules(const std::string &metadata, std::vector<Residue*> residues, FileType fileType){
	std::cout<<"---------------------Molecules---------------------" <<std::endl;
	time_t startTime = time(nullptr);
	int index = 0;
	std::istringstream cmpnd(metadata);
	std::string line;
	std::vector<Molecule*> molecules;
	std::map<char, int> molChains;
  if(fileType == PDB){
    while(std::getline(cmpnd, line)){
  		if(line.substr(0,6) != "COMPND") continue;
  		if(line.find("MOL_ID") != std::string::npos){
  			std::getline(cmpnd, line);
  			if(line.find(";") == std::string::npos){
  				line = line.substr(21);
  				trim(line);
  				std::string temp;
  				std::getline(cmpnd, temp);
  				line += temp.substr(11, temp.find_first_of(';') - 11);
  				trim(line);
  			}
  			else{
  				line = line.substr(21, line.find_first_of(';') - 21);
  				trim(line);
  			}
  			const char* moleculeName = line.c_str();
  			Molecule* mol = new Molecule(index);
        mol->name = moleculeName;
  			molecules.push_back(mol);
  		}
  		else if(line.find("CHAIN:") != std::string::npos){
  			line = line.substr(18);
  			if(line.find(",") != std::string::npos){
  				for(unsigned int i = 0; i < line.length() && line.at(i) != ';'; i+=3){
  					molChains.insert(std::make_pair(line.at(i),index));
  				}
  			}
  			else{
  				molChains.insert(std::make_pair(line.at(0),index));
  			}
  			index++;
  		}
  	}
  }
  else{
    throw UnsupportedFileType("ERROR this file type is not supported yet");
  }

  for(auto res = residues.begin(); res != residues.end(); ++res){
    molecules[molChains[(*res)->chain]]->residues.push_back((*res));
  }

	std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for separation of molecules."<<std::endl;
  return molecules;
}

void parsePDB(std::string pathToFile, std::vector<Molecule*> &molecules, std::string &metadata, std::string &identifier, std::string &extraInfo){
  std::cout<<"----------------------Parsing-----------------------" <<std::endl;
  identifier = pathToFile.substr(pathToFile.length() - 8, 4);
	std::ifstream pdbstream(pathToFile);
	std::string currentLine;
	bool stillHeader = true;
  int model = 0;
  std::vector<Residue*> allResidues;
  std::string resChecker;
	if (pdbstream.is_open()) {
		while (getline(pdbstream, currentLine)) {
			if (currentLine.substr(0, 5) == "MODEL"){
        if(++model > 1) break;
      }
			else if ((currentLine.substr(0, 4) == "ATOM" || (currentLine.substr(0, 6) == "HETATM")) && currentLine.substr(17,3) != "HOH") {
        stillHeader = false;

        Atom* currentAtom = new Atom(currentLine, PDB);
        if(allResidues.size() != 0){
          resChecker = std::to_string(allResidues[allResidues.size() - 1]->id);
          if(allResidues[allResidues.size() - 1]->insertion != '-'){
            resChecker += allResidues[allResidues.size() - 1]->insertion;
          }
          else{
            resChecker += " ";
          }
          while(resChecker.length() < 5) resChecker = " " + resChecker;
        }
        if(allResidues.size() == 0 || resChecker != currentLine.substr(22,5)){
          allResidues.push_back(new Residue(stoi(currentLine.substr(22, 4))));
          allResidues[allResidues.size() - 1]->name = currentLine.substr(17, 4);
          allResidues[allResidues.size() - 1]->chain = currentLine.at(21);
          allResidues[allResidues.size() - 1]->insertion = currentLine.at(26);
          trim( allResidues[allResidues.size() - 1]->name);
          if( allResidues[allResidues.size() - 1]->name.length() == 0){
             allResidues[allResidues.size() - 1]->name = currentAtom->element;
          }
        }
        currentAtom->parent = allResidues[allResidues.size() - 1];

        if(currentAtom->altLoc == ' ' || currentAtom->altLoc == '1' || currentAtom->altLoc == 'A'){
          allResidues[allResidues.size() - 1]->atoms.push_back(currentAtom);
        }
        else{
          delete currentAtom;
          continue;
        }
			}
			else if(stillHeader){
				metadata += currentLine + "\n";
			}
      else{
        extraInfo += currentLine + "\n";
      }
		}
		pdbstream.close();
    std::cout<<"done parsing "<<pathToFile<<std::endl;
    molecules = separateMolecules(metadata, allResidues, PDB);
  }
	else std::cout<< "Unable to open: " + pathToFile <<std::endl;
}
void parsePDB(std::string pathToFile, std::vector<Residue*> &residues){
  std::cout<<"----------------------Parsing-----------------------" <<std::endl;
	std::ifstream pdbstream(pathToFile);
	std::string currentLine;
  int model = 0;
  std::string resChecker;
	if (pdbstream.is_open()) {
		while (getline(pdbstream, currentLine)) {
      if (currentLine.substr(0, 5) == "MODEL"){
        if(++model > 1) break;
      }
			else if ((currentLine.substr(0, 4) == "ATOM" || (currentLine.substr(0, 6) == "HETATM")) && currentLine.substr(17,3) != "HOH") {
        Atom* currentAtom = new Atom(currentLine, PDB);
        if(residues.size() != 0){
          resChecker = std::to_string(residues[residues.size() - 1]->id);
          if(residues[residues.size() - 1]->insertion != '-'){
            resChecker += residues[residues.size() - 1]->insertion;
          }
          else{
            resChecker += " ";
          }
          while(resChecker.length() < 5) resChecker = " " + resChecker;
        }
        if(residues.size() == 0 || resChecker != currentLine.substr(22,5)){
          residues.push_back(new Residue(stoi(currentLine.substr(22, 4))));
          residues[residues.size() - 1]->name = currentLine.substr(17, 4);
          residues[residues.size() - 1]->chain = currentLine.at(21);
          residues[residues.size() - 1]->insertion = currentLine.at(26);
          trim( residues[residues.size() - 1]->name);
          if( residues[residues.size() - 1]->name.length() == 0){
             residues[residues.size() - 1]->name = currentAtom->element;
          }
        }
        currentAtom->parent = residues[residues.size() - 1];

        if(currentAtom->altLoc == ' ' || currentAtom->altLoc == '1' || currentAtom->altLoc == 'A'){
          residues[residues.size() - 1]->atoms.push_back(currentAtom);
        }
        else{
          delete currentAtom;
          continue;
        }
			}
		}
		pdbstream.close();
    std::cout<<"done parsing "<<pathToFile<<std::endl;
  }
	else std::cout<< "Unable to open: " + pathToFile <<std::endl;
}

std::string createMolecularJSON(const std::vector<Molecule*> &molecules, const std::string &identifier){
  std::string jsonOutput = "";
  json::JSON fullJSON;
  fullJSON["identifier"] = identifier;
  fullJSON["molecules"] = json::Array();
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    json::JSON currentMol;
    currentMol["identifier"] = identifier;
    currentMol["mol_id"] = (*mol)->id;
    currentMol["mol_name"] = (*mol)->name;
    switch((*mol)->type){
      case protein:
        currentMol["mol_type"] = "protein";
        break;
      case ligand:
        currentMol["mol_type"] = "ligand";
        break;
      case dna:
        currentMol["mol_type"] = "dna";
        break;
      case rna:
        currentMol["mol_type"] = "rna";
        break;
      case carbohydrate:
        currentMol["mol_type"] = "carbohydrate";
        break;
      default:
        currentMol["mol_type"] = "unknown";
        break;
    }
    currentMol["residues"] = json::Array();
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      json::JSON currentRes;
      switch((*res)->type){
        case protein:
          currentRes["res_type"] = "protein";
          break;
        case ligand:
          currentRes["res_type"] = "ligand";
          break;
        case dna:
          currentRes["res_type"] = "dna";
          break;
        case rna:
          currentRes["res_type"] = "rna";
          break;
        case carbohydrate:
          currentRes["res_type"] = "carbohydrate";
          break;
        default:
          currentRes["res_type"] = "unknown";
          break;
      }
      currentRes["res_id"] = (*res)->id;
      currentRes["chain"] = (*res)->chain;
      currentRes["insertion"] = (*res)->insertion;
      currentRes["res_name"] = (*res)->name;
      currentRes["relative_affinity"] = (*res)->relativeAffinity;
      currentRes["atoms"] = json::Array();
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        json::JSON currentAtom;
        currentAtom["atom_id"] = (*atom)->id;
        currentAtom["element"] = (*atom)->element;
        currentAtom["atom_name"] = (*atom)->name;
        currentAtom["beta_factor"] = (*atom)->betaFactor;
        currentAtom["charge"] = (*atom)->charge;
        currentAtom["atom_type"] = (*atom)->type;
        currentAtom["relative_affinity"] = (*atom)->relativeAffinity;
        currentAtom["true_positive"] = (*atom)->truePositive;
        currentAtom["x"] = (*atom)->sphere.center.x;
        currentAtom["y"] = (*atom)->sphere.center.y;
        currentAtom["z"] = (*atom)->sphere.center.z;
        currentAtom["surf"] = (*atom)->sphere.surf;
        currentAtom["relative_affinity"] = (*atom)->relativeAffinity;
        currentRes["atoms"].append(currentAtom);
      }
      currentMol["residues"].append(currentRes);
    }
    fullJSON["molecules"].append(currentMol);
  }
  jsonOutput += fullJSON.dump();
  std::cout << identifier  << " string has been created." << std::endl;
  return jsonOutput;
}
void readMolecularJSON(std::string jsonStr, std::vector<Molecule*> &molecules, std::string &identifier){
  json::JSON fullJSON;
  size_t offvector = 0;
  fullJSON = json::parse_object(jsonStr, offvector);
  std::string typeHelper;
  std::map<std::string, MoleculeType> strToType;
  strToType["protein"] = protein;
  strToType["ligand"] = ligand;
  strToType["dna"] = dna;
  strToType["rna"] = rna;
  strToType["carbohydrate"] = carbohydrate;
  int numMolecules = fullJSON.at("molecules").length();
  for(int m = 0; m < numMolecules; ++m){
    Molecule* molecule = new Molecule();
    molecule->id = fullJSON.at("molecules")[m].at("mol_id").ToInt();
    molecule->name = fullJSON.at("molecules")[m].at("mol_name").ToString();
    molecule->type = strToType[fullJSON.at("molecules")[m].at("mol_type").ToString()];
    int numResidues = fullJSON.at("molecules")[m].at("residues").length();
    for(int r = 0; r < numResidues; ++r){
      Residue* residue = new Residue();
      residue->id = fullJSON.at("molecules")[m].at("residues")[r].at("res_id").ToInt();
      residue->chain = static_cast<char>(fullJSON.at("molecules")[m].at("residues")[r].at("chain").ToInt());
      residue->insertion = static_cast<char>(fullJSON.at("molecules")[m].at("residues")[r].at("insertion").ToInt());
      residue->name = fullJSON.at("molecules")[m].at("residues")[r].at("res_name").ToString();
      residue->type = strToType[fullJSON.at("molecules")[m].at("residues")[r].at("res_type").ToString()];
      residue->relativeAffinity = fullJSON.at("molecules")[m].at("residues")[r].at("relative_affinity").ToFloat();
      int numAtoms = fullJSON.at("molecules")[m].at("residues")[r].at("atoms").length();
      // std::cout << "Num Atoms: " << numAtoms << std::endl;
      for(int a = 0; a < numAtoms; ++a){
        Atom* atom = new Atom();
        atom->id = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("atom_id").ToInt();
        atom->element = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("element").ToString();
        atom->name = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("atom_name").ToString();
        atom->betaFactor = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("beta_factor").ToFloat();
        atom->charge = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("charge").ToInt();
        atom->type = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("atom_type").ToString();
        if(atom->type != "---") updateAtomTypeMap(atom->type);
        atom->relativeAffinity = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("relative_affinity").ToFloat();
        atom->truePositive = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("true_positive").ToBool();
        atom->sphere.center.x = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("x").ToFloat();
        atom->sphere.center.y = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("y").ToFloat();
        atom->sphere.center.z = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("z").ToFloat();
        atom->sphere.radius = radius[atom->element];
        atom->sphere.molResAtom = {m,r,a};
        atom->sphere.surf = fullJSON.at("molecules")[m].at("residues")[r].at("atoms")[a].at("surf").ToBool() == 1;
        atom->parent = residue;
        residue->atoms.push_back(atom);
      }
      residue->parent = molecule;
      molecule->residues.push_back(residue);
    }
    molecules.push_back(molecule);
  }
  identifier = fullJSON.at("identifier").ToString();
}

std::string createTrianglesJSON(const std::vector<Triangle> &triangles){
  std::string jsonOutput = "";
  json::JSON trianglesJSON;
  trianglesJSON["triangles"] = json::Array();
  int id = 0;
  for (auto tri = triangles.begin(); tri != triangles.end(); ++tri) {
    json::JSON currentTriangle;
    currentTriangle["id"] = id++;
    currentTriangle["atom_type_1"] = (*tri).atomTypes.x;
    currentTriangle["atom_type_2"] = (*tri).atomTypes.y;
    currentTriangle["atom_type_3"] = (*tri).atomTypes.z;
    currentTriangle["occurances"] = (*tri).occurances;
    currentTriangle["interactions"] = (*tri).interactions;
    currentTriangle["affinity"] = (*tri).affinity;
    trianglesJSON["triangles"].append(currentTriangle);
  }
  jsonOutput += trianglesJSON.dump();
  return jsonOutput;
}
void readTrianglesJSON(std::string triangleStr, std::vector<Triangle> &triangles){
  json::JSON trianglesJSON;
  size_t offvector = 0;
  trianglesJSON = json::parse_object(triangleStr, offvector);
  int trianglesLength = trianglesJSON.at("triangles").length();
  for (int i = 0; i < trianglesLength; ++i) {
    Triangle currentTriangle;
    currentTriangle.occurances = trianglesJSON.at("triangles")[i].at("occurances").ToFloat();
    currentTriangle.interactions = trianglesJSON.at("triangles")[i].at("interactions").ToFloat();
    currentTriangle.affinity = trianglesJSON.at("triangles")[i].at("affinity").ToFloat();
    currentTriangle.atomTypes.x = trianglesJSON.at("triangles")[i].at("atom_type_1").ToInt();
    currentTriangle.atomTypes.y = trianglesJSON.at("triangles")[i].at("atom_type_2").ToInt();
    currentTriangle.atomTypes.z = trianglesJSON.at("triangles")[i].at("atom_type_3").ToInt();
    triangles.push_back(currentTriangle);
  }
}


void prepareVisualizationFile(const std::vector<Molecule*> &molecules, std::string identifier, bool residueScoring){
  std::ofstream outstream("data/sites/" + identifier + "_site.pdb");
  std::string currentLine;
  float currentScore = 0.0f;
  float max = 0.0f;
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      if(residueScoring){
        currentScore = (*res)->relativeAffinity;
        if(currentScore > max) max = currentScore;
        continue;
      }
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        currentScore = (*atom)->relativeAffinity;
        if(currentScore > max) max = currentScore;
      }
    }
  }

  float diviser = 1.0f;
  while(max/diviser >= 1000.0f) diviser *= 10.0f;

  if (outstream.is_open()) {
    int serialNum = 0;
    for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
      for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
        for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
          currentScore = (residueScoring) ? (*res)->relativeAffinity : (*atom)->relativeAffinity;
          if(std::isnan(currentScore)){
            std::cout<<"ERROR BAD SCORE: nan"<<std::endl;
            exit(-1);
          }
          currentLine = (*atom)->createPDBDescriptor(serialNum++, 100.0f*(currentScore/max)) + "\n";
          outstream << currentLine;
        }
      }
    }
    outstream.close();
    std::cout<<"./data/sites/" + identifier + "_site.pdb has been created"<<std::endl;
  }
  else{
    std::cout<<"ERROR creating bsp visualization"<<std::endl;
  }
}
void prepareVisualizationFile(const std::vector<Molecule*> &molecules, std::string identifier, MoleculeType type, bool residueScoring){
  std::string molType;
  switch(type){
    case protein:
      molType = "protein";
      break;
    case ligand:
      molType = "ligand";
      break;
    case dna:
      molType = "dna";
      break;
    case rna:
      molType = "rna";
      break;
    case carbohydrate:
      molType = "carbohydrate";
      break;
    default:
      molType = "unknown";
      break;
  }
  std::ofstream outstream("data/sites/" + identifier + "_" + molType + "_site.pdb");
  std::string currentLine;
  float currentScore = 0.0f;
  float max = 0.0f;
  for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
    for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
      if(residueScoring){
        currentScore = (*res)->relativeAffinity;
        if(currentScore > max) max = currentScore;
        continue;
      }
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        currentScore = (*atom)->relativeAffinity;
        if(currentScore > max) max = currentScore;
      }
    }
  }

  float diviser = 1.0f;
  while(max/diviser >= 1000.0f) diviser *= 10.0f;

	if (outstream.is_open()) {
    int serialNum = 0;
    for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
      if((*mol)->type != protein) continue;
      for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
        for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
          currentScore = (residueScoring) ? (*res)->relativeAffinity : (*atom)->relativeAffinity;
          if(std::isnan(currentScore)){
            std::cout<<"ERROR BAD SCORE: nan"<<std::endl;
            exit(-1);
          }
          currentLine = (*atom)->createPDBDescriptor(serialNum++, 100.0f*(currentScore/max)) + "\n";
          outstream << currentLine;
        }
      }
    }

		outstream.close();
    std::cout<<"./data/sites/" + identifier + "_" + molType + "_site.pdb has been created"<<std::endl;
  }
  else{
    std::cout<<"ERROR creating bsp visualization"<<std::endl;
  }
}
void prepareVisualizationFile(const Molecule* molecule, std::string identifier, bool residueScoring){
  std::ofstream outstream("data/sites/" + identifier  + "_" + molecule->name + "_site.pdb");
  std::string currentLine;
  float currentScore = 0.0f;
  float max = 0.0f;
  for(auto res = molecule->residues.begin(); res != molecule->residues.end(); ++res){
    if(residueScoring){
      currentScore = (*res)->relativeAffinity;
      if(currentScore > max) max = currentScore;
      continue;
    }
    for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
      currentScore = (*atom)->relativeAffinity;
      if(currentScore > max) max = currentScore;
    }
  }

  float diviser = 1.0f;
  while(max/diviser >= 1000.0f) diviser *= 10.0f;

  if (outstream.is_open()) {
    int serialNum = 0;
    for(auto res = molecule->residues.begin(); res != molecule->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        currentScore = (residueScoring) ? (*res)->relativeAffinity : (*atom)->relativeAffinity;
        if(std::isnan(currentScore)){
          std::cout<<"ERROR BAD SCORE: nan"<<std::endl;
          exit(-1);
        }
        currentLine = (*atom)->createPDBDescriptor(serialNum++, 100.0f*(currentScore/max)) + "\n";
        outstream << currentLine;
      }
    }
    outstream.close();
    std::cout<<"./data/sites/" + identifier + "_site.pdb has been created"<<std::endl;
  }
  else{
    std::cout<<"ERROR creating bsp visualization"<<std::endl;
  }
}
void writeAtomTypeChecker(const std::vector<Molecule*> &molecules, const std::string &identifier){
  std::string pathToFile = "data/atomTypeCheckers/" + identifier + "_atomtype" + ".pdb";
  std::ofstream outstream(pathToFile);
	std::string currentLine;
  std::string temp;
	if(outstream.is_open()) {
    for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
      for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
        for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
          currentLine = ((*atom)->element.length() < 2) ? (*atom)->element + " " : (*atom)->element;
          temp = (*atom)->name;
          while(temp.length() < 4) temp += " ";
          currentLine += temp + " ";
          temp = (*res)->name;
          while(temp.length() < 3) temp += " ";
          currentLine += temp + " ";
          currentLine += std::to_string((*res)->chain) + " ";
          temp = std::to_string((*res)->id);
          while(temp.length() < 3) temp += " ";
          currentLine += temp + " ";
          currentLine += (*atom)->type + "\n";
          outstream << currentLine;
        }
      }
    }
    std::cout<< pathToFile + " has been created." <<std::endl;
  }
	else{
		std::cout<<"error creating "<<pathToFile<<std::endl;
	}
}
void writeAtomTypeChecker(const Molecule* molecule, const std::string &identifier){
  std::string pathToFile = "data/atomTypeCheckers/" + identifier + "_" + molecule->name + "_atomtype" + ".pdb";
  std::ofstream outstream(pathToFile);
	std::string currentLine;
  std::string temp;
	if(outstream.is_open()) {
    for(auto res = molecule->residues.begin(); res != molecule->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        currentLine = ((*atom)->element.length() < 2) ? (*atom)->element + " " : (*atom)->element;
        temp = (*atom)->name;
        while(temp.length() < 4) temp += " ";
        currentLine += temp + " ";
        temp = (*res)->name;
        while(temp.length() < 3) temp += " ";
        currentLine += temp + " ";
        currentLine += std::to_string((*res)->chain) + " ";
        temp = std::to_string((*res)->id);
        while(temp.length() < 3) temp += " ";
        currentLine += temp + " ";
        currentLine += (*atom)->type + "\n";
        outstream << currentLine;
      }
    }
    std::cout<< pathToFile + " has been created."<<std::endl;
  }
	else{
		std::cout<<"error creating "<<pathToFile<<std::endl;
	}
}
void writeSurfaceChecker(const std::vector<Molecule*> &molecules, const std::string &identifier){
  std::ofstream outstream("./data/surfaceCheckers/" + identifier + "_surf" + ".pdb");
	std::string tempFactor;
  std::string currentLine;
	if(outstream.is_open()) {
    int serialNum = 0;
    for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
      for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
        for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
          if((*atom)->sphere.surf){
            currentLine = (*atom)->createPDBDescriptor(serialNum++, 100.0f) + "\n";
          }
          else{
            currentLine = (*atom)->createPDBDescriptor(serialNum++, 0.0f) + "\n";
          }
          outstream << currentLine;
        }
      }
    }
		outstream.close();
		std::cout<< identifier +".surfaceChecker" + " surface checker has been created." <<std::endl;
	}
  else{
    std::cout<<"ERROR cannot create "<<"./data/surfaceCheckers/" + identifier + "_surf" + ".pdb"<<std::endl;
    exit(-1);
  }
}
void writeSurfaceChecker(const Molecule* molecule, std::string identifier){
  std::ofstream outstream("./data/surfaceCheckers/" + identifier + "_" + molecule->name + "_surf" + ".pdb");
	std::string tempFactor;
  std::string currentLine;
	if(outstream.is_open()) {
    int serialNum = 0;
    for(auto res = molecule->residues.begin(); res != molecule->residues.end(); ++res){
      for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
        if((*atom)->sphere.surf){
          currentLine = (*atom)->createPDBDescriptor(serialNum++, 100.0f) + "\n";
        }
        else{
          currentLine = (*atom)->createPDBDescriptor(serialNum++, 0.0f) + "\n";
        }
        outstream << currentLine;
      }
    }
		outstream.close();
		std::cout<< identifier +".surfaceChecker" + " surface checker has been created." <<std::endl;
	}
  else{
    std::cout<<"ERROR cannot create "<<"./data/surfaceCheckers/" + identifier + "_" + molecule->name + "_surf" + ".pdb"<<std::endl;
    exit(-1);
  }
}
void writeTrainingLog(std::string identifier, std::vector<Triangle> triangles, std::vector<std::string> datavector){
  std::string file = "./data/"+identifier+"_trainingLog.txt";
  std::ofstream outstream(file);
  std::string currentLine = "";
  if(outstream.is_open()){
    std::cout<<"writing "<< "./data/"<<identifier<<"_trainingLog.txt"<<std::endl;
  }
  else{
    std::cout<<"cannot write "<< "./data/"<<identifier<<"_trainingLog.txt"<<std::endl;
    exit(-1);
  }
  currentLine = "========================TRIANGLES========================\n";
  int temp = 0;
  outstream<<currentLine;
  for(auto tri = triangles.begin(); tri != triangles.end(); ++tri){
    if((*tri).occurances == 0.0f) continue;
    ++temp;
    outstream<<(*tri).occurances<<" " << (*tri).interactions<<" "<<(*tri).affinity<<" - (";

    std::string at1;
    std::string at2;
    std::string at3;
    bool found[3] = {false};
    for(auto &i : atomTypeMap){
      if(i.second == (*tri).atomTypes.x){
        at1 = i.first;
        found[0] = true;
      }
      if(i.second == (*tri).atomTypes.y){
        at2 = i.first;
        found[1] = true;
      }
      if(i.second == (*tri).atomTypes.z){
        at3 = i.first;
        found[2] = true;
      }
      if(found[0] && found[1] && found[2]){
        break;
      }
    }

    outstream << at1 <<"," << at2 << "," << at3 << ")\n";
  }
  outstream<<"\nnumber of triangle types found= "<<temp<<"\n";

  outstream<<"training vector size = "<<datavector.size()<<"\n";
  currentLine = "\n=========================identifierS=========================\n";
  outstream<<currentLine;
  for(auto str = datavector.begin(); str !=  datavector.end(); ++str){
    outstream << *str << "\n";
  }
}
