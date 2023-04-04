void checkSurfaceEquivalency(std::string pathToPDB, int numPoints, const std::vector<Residue*> &residues){
  int numFalsePositives = 0;
  int numFalseNegatives = 0;
  int numCorrect = 0;
  int numAtoms = 0;
  std::string pathToFreeSASAPDB = pathToPDB.substr(0,pathToPDB.find_first_of('.')) + "_freesasa.pdb";

  std::string freesasaCMD = "freesasa --shrake-rupley -n " + std::to_string(numPoints);
  freesasaCMD += " --depth=atom -f pdb < " + pathToPDB + " > " + pathToFreeSASAPDB;

  system(freesasaCMD.c_str());

  std::vector<Residue*> freeSASAComplex;
  std::string metadata;
  std::string identifier = "sasa_checker";
  std::string extraInfo;

  std::cout<<pathToFreeSASAPDB<<std::endl;

  parsePDB(pathToFreeSASAPDB, freeSASAComplex);
  int3 molResAtom = {0,0,0};

  for(auto res = freeSASAComplex.begin(); res != freeSASAComplex.end(); ++res, ++molResAtom.y){
    for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom, ++molResAtom.z){
      (*atom)->sphere.molResAtom = molResAtom;
      if((*atom)->betaFactor > 0.0f){
        (*atom)->sphere.surf = true;
      }
      else{
        (*atom)->sphere.surf = false;
      }
    }
  }
  if(freeSASAComplex.size() != residues.size()){
    std::cout<<"ERROR free sasa and original pdb have different number of residues "<<freeSASAComplex.size()<<" "<<residues.size()<<std::endl;
  }
  for(unsigned int r = 0; r < freeSASAComplex.size(); ++r){
    if(freeSASAComplex[r]->atoms.size() != residues[r]->atoms.size()){
      std::cout<<"ERROR free sasa residue and original pdb residue do not have the same number of atoms "<<freeSASAComplex[r]->atoms.size()<<" "<<residues[r]->atoms.size()<<std::endl;
      exit(-1);
    }
    for(unsigned int a = 0; a < freeSASAComplex[r]->atoms.size(); ++a){
      //check to see if same atom
      if(residues[r]->atoms[a]->id != freeSASAComplex[r]->atoms[a]->id){
        std::cout<<"ERROR wrong atom in surface confirmation"<<std::endl;
        exit(-1);
      }
      if(residues[r]->atoms[a]->sphere.surf == freeSASAComplex[r]->atoms[a]->sphere.surf){
        ++numCorrect;
      }
      else if(residues[r]->atoms[a]->sphere.surf){
        ++numFalsePositives;
      }
      else{
        ++numFalseNegatives;
      }
      ++numAtoms;
    }void checkSurfaceEquivalency_getArea(std::string pathToGetAreaFile, const std::vector<Molecule*> &molecules){
  std::ifstream gastream(pathToGetAreaFile);
  std::string currentLine;
  std::map<int, float> ga_info;
  int numFalsePositives = 0;
  int numFalseNegatives = 0;
  int numCorrect = 0;
  int numAtoms = 0;
  if (gastream.is_open()) {
    while (getline(gastream, currentLine)) {
      if(currentLine == "") continue;
      ga_info.insert(std::make_pair(std::stoi(currentLine.substr(0,6)), std::stof(currentLine.substr(21,8))));
      ++numAtoms;
    }
    for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
      for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
        for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
          if(ga_info[(*atom)->id] == 0.0f){
            if(!(*atom)->sphere.surf) ++numCorrect;
            else{
              ++numFalsePositives;
            }
          }
          else{
            if((*atom)->sphere.surf) ++numCorrect;
            else{
              ++numFalseNegatives;
            }
          }
        }
      }
    }
    gastream.close();
    printf("numCorrect = %d, numFalseNegatives %d, numFalsePositives %d\n",numCorrect,numFalseNegatives,numFalsePositives);
    std::cout<<(float)numCorrect*100/numAtoms<<"% correct"<<std::endl;
    std::cout<<(float)numFalseNegatives*100/numAtoms<<"% false negative"<<std::endl;
    std::cout<<(float)numFalsePositives*100/numAtoms<<"% false positive"<<std::endl;
  }
}


  }


  printf("numCorrect = %d, numFalseNegatives %d, numFalsePositives %d\n",numCorrect,numFalseNegatives,numFalsePositives);
  std::cout<<(float)numCorrect*100/numAtoms<<"% correct"<<std::endl;
  std::cout<<(float)numFalseNegatives*100/numAtoms<<"% false negative"<<std::endl;
  std::cout<<(float)numFalsePositives*100/numAtoms<<"% false positive"<<std::endl;
}

void checkSurfaceEquivalency_getArea(std::string pathToGetAreaFile, const std::vector<Molecule*> &molecules){
  std::ifstream gastream(pathToGetAreaFile);
  std::string currentLine;
  std::map<int, float> ga_info;
  int numFalsePositives = 0;
  int numFalseNegatives = 0;
  int numCorrect = 0;
  int numAtoms = 0;
  if (gastream.is_open()) {
    while (getline(gastream, currentLine)) {
      if(currentLine == "") continue;
      ga_info.insert(std::make_pair(std::stoi(currentLine.substr(0,6)), std::stof(currentLine.substr(21,8))));
      ++numAtoms;
    }
    for(auto mol = molecules.begin(); mol != molecules.end(); ++mol){
      for(auto res = (*mol)->residues.begin(); res != (*mol)->residues.end(); ++res){
        for(auto atom = (*res)->atoms.begin(); atom != (*res)->atoms.end(); ++atom){
          if(ga_info[(*atom)->id] == 0.0f){
            if(!(*atom)->sphere.surf) ++numCorrect;
            else{
              ++numFalsePositives;
            }
          }
          else{
            if((*atom)->sphere.surf) ++numCorrect;
            else{
              ++numFalseNegatives;
            }
          }
        }
      }
    }
    gastream.close();
    printf("numCorrect = %d, numFalseNegatives %d, numFalsePositives %d\n",numCorrect,numFalseNegatives,numFalsePositives);
    std::cout<<(float)numCorrect*100/numAtoms<<"% correct"<<std::endl;
    std::cout<<(float)numFalseNegatives*100/numAtoms<<"% false negative"<<std::endl;
    std::cout<<(float)numFalsePositives*100/numAtoms<<"% false positive"<<std::endl;
  }
}
