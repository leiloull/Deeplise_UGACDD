#include "Triangle.cuh"

Triangle::Triangle() {
    this->affinity = 0.0f;
    this->occurances = 0;
    this->interactions = 0;
    this->atomTypes = {-1,-1,-1};
}
Triangle::Triangle(int3 atomTypes) {
    this->atomTypes = atomTypes;
    this->affinity = 0.0f;
    this->occurances = 0;
    this->interactions = 0;
}
void Triangle::printTriangle(){


  std::cout<<"("<<this->affinity<<" " << this->interactions<<" "<< this->occurances<<" ";

  std::string at1;
  std::string at2;
  std::string at3;
  bool found[3] = {false};
  for(auto &i : atomTypeMap){
    if(i.second == this->atomTypes.x){
      at1 = i.first;
      found[0] = true;
    }
    if(i.second == this->atomTypes.y){
      at2 = i.first;
      found[1] = true;
    }
    if(i.second == this->atomTypes.z){
      at3 = i.first;
      found[2] = true;
    }
    if(found[0] && found[1] && found[2]){
      break;
    }
  }

  std::cout<< at1 <<"," << at2 << "," << at3 << ")" << std::endl;
}
bool compareByOccurances(const Triangle &a, const Triangle &b){
  return a.occurances > b.occurances;
}
bool compareByInteractions(const Triangle &a, const Triangle &b){
  return a.interactions > b.interactions;
}
bool compareByAffinity(const Triangle &a, const Triangle &b){
  return a.affinity > b.affinity;
}
