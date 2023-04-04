#ifndef BIO_MAPS_H
#define BIO_MAPS_H

#include "common_includes.h"

extern std::map<std::string, float> radius;
extern std::map<int, char> moleculeTypeMap;
extern std::map<std::string, unsigned char> atomTypeMap;
extern std::map<std::string, unsigned char> IgnoreListMap;

void updateAtomTypeMap(std::string atomType);
void updateIgnoreListMap(std::string atomType);

#endif /* BIO_MAPS_H */
