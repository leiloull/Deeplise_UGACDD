#include "bio_maps.h"

//test comment

std::map<std::string, float> radius = {
  {"H",1.20},
  {"HE",1.40},
  {"LI",1.82},
  {"BE",1.53},
  {"B",1.92},
  {"C",1.70},
  {"N",1.55},
  {"O",1.52},
  {"F",1.47},
  {"NE",1.54},
  {"NA",2.27},
  {"MG",1.73},
  {"AL",1.84},
  {"SI",2.10},
  {"P",1.80},
  {"S",1.80},
  {"CL",1.75},
  {"AR",1.88},
  {"K",2.75},
  {"CA",2.31},
  {"SC",2.11},
  {"TI",1.76}, //None observed
  {"V",1.71},  //None observed
  {"CR",1.66}, //None observed
  {"MN",1.61}, //None observed
  {"FE",1.56}, //None observed
  {"CO",1.52}, //None observed
  {"NI",1.63},
  {"CU",1.40},
  {"ZN",1.39},
  {"GA",1.87},
  {"GE",2.11},
  {"AS",1.85},
  {"SE",1.90},
  {"BR",1.85},
  {"KR",2.02},
  {"RB",3.03},
  {"SR",2.49},
  {"Y",2.12},   //None observed
  {"ZR",2.06},  //None observed
  {"NB",1.98},  //None observed
  {"MO",1.90},  //None observed
  {"TC",1.83},  //None observed
  {"RU",1.78},  //None observed
  {"RH",1.73},  //None observed
  {"PD",1.63},
  {"AG",1.72},
  {"CD",1.58},
  {"IN",1.93},
  {"SN",2.17},
  {"SB",2.06},
  {"TE",2.06},
  {"I",1.98},
  {"XE",2.16},
  {"CS",3.43},
  {"BA",2.68},
  {"LA",1.95},  //None observed
  {"CE",1.58},  //None observed
  {"PR",2.47},  //None observed
  {"ND",2.06},  //None observed
  {"PM",2.05},  //None observed
  {"SM",2.38},  //None observed
  {"EU",2.31},  //None observed
  {"GD",2.33},  //None observed
  {"TB",2.25},  //None observed
  {"DY",2.28},  //None observed
  {"HO",2.26},  //None observed
  {"ER",2.26},  //None observed
  {"TM",2.22},  //None observed
  {"YB",2.22},  //None observed
  {"LU",2.17},  //None observed
  {"HF",2.08},  //None observed
  {"TA",2.00},  //None observed
  {"W",1.93},   //None observed
  {"RE",1.88},  //None observed
  {"OS",1.85},  //None observed
  {"IR",1.80},  //None observed
  {"PT",1.75},
  {"AU",1.66},
  {"HG",1.55},
  {"TI",1.96},
  {"PB",2.02},
  {"BI",2.07},
  {"PO",1.97},
  {"AT",2.02},
  {"RN",2.20}
};

// std::map<std::string, float> radius = {
//   {"C",0.67},
//   {"O",0.48},
//   {"H",0.53},
//   {"D",0.53},
//   {"N",0.56},
//   {"P",0.98},
//   {"F",0.42},
//   {"NA",1.9},
//   {"MG",1.45},
//   {"Si",1.11},
//   {"S",0.88},
//   {"CL",0.79},
//   {"K",2.43},
//   {"CA",1.94},
//   {"MN",1.61},
//   {"FE",1.56},
//   {"CO",1.52},
//   {"NI",1.49},
//   {"CU",1.45},
//   {"ZN",1.42},
//   {"BR",0.94},
//   {"AG",1.65},
//   {"SN",1.45},
//   {"PT",1.77},
//   {"AU",1.74},
//   {"PB",1.54},
//   {"HG",1.71}
// };

std::map<int, char> moleculeTypeMap = {
  {0, 'P'},
  {1, 'L'},
  {2, 'D'},
  {3, 'R'},
  {4, 'C'},
  {5, 'U'}
};
std::map<std::string, unsigned char> atomTypeMap;
void updateAtomTypeMap(std::string atomType){
  if(atomType == "---") return;
  //std::cout<<(*atomTypeMap.begin()).first<<std::endl;
	int mapSize = atomTypeMap.size();
  auto search = atomTypeMap.find(atomType);
  if(search == atomTypeMap.end()){
    atomTypeMap.insert(std::make_pair(atomType, mapSize));
    std::cout<<atomType<<" added to atomTypeMap as "<<mapSize<<std::endl;
  }
}

std::map<std::string, unsigned char> IgnoreListMap;
void updateIgnoreListMap(std::string atomType){
  if(atomType == "---") return;
  //std::cout<<(*atomTypeMap.begin()).first<<std::endl;
	int mapSize = IgnoreListMap.size();
  auto search = IgnoreListMap.find(atomType);
  if(search == IgnoreListMap.end()){
    IgnoreListMap.insert(std::make_pair(atomType, mapSize));
    std::cout<<atomType<<" added to IgnoreListMap as "<<mapSize<<std::endl;
  }
}