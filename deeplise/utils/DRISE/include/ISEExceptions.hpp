#ifndef ISEEXCEPTION_HPP
#define ISEEXCEPTION_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>

struct ISEException : public std::exception{
  std::string msg;
  ISEException(){
    msg = "Unknown ISE error";
  }
  ISEException(std::string msg) : msg("Unknown ISE error: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct ISEException_logic : public std::logic_error{
  std::string msg;
  ISEException_logic() : std::logic_error("ISE logic error"){
    msg = "ISE logic error";
  }
  ISEException_logic(std::string msg) : std::logic_error(msg), msg("ISE logic error: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct ISEException_runtime : public std::runtime_error{
  std::string msg;
  ISEException_runtime() : std::runtime_error("ISE runtime error"){
    msg = "ISE runtime error";
  }
  ISEException_runtime(std::string msg) : std::runtime_error(msg), msg("ISE runtime error: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct ComplexSpecificException : public ISEException_runtime{
  std::string msg;
  ComplexSpecificException(){
    msg = "ISE Single Complex Exception";
  }
  ComplexSpecificException(std::string msg) : msg("ISE Single Complex Exception: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct PDBFormatException : public ComplexSpecificException{
  std::string msg;
  PDBFormatException(){
    msg = "Error in pdb formating";
  }
  PDBFormatException(std::string msg) : msg("Error in pdb formating: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct UnsupportedFileType : public ComplexSpecificException{
  std::string msg;
  UnsupportedFileType(){
    msg = "Unsupported File Type";
  }
  UnsupportedFileType(std::string msg) : msg("File type not supported: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct MissingAtomType : public ComplexSpecificException{
  std::string msg;
  MissingAtomType(){
    msg = "Cannot determine atom type";
  }
  MissingAtomType(std::string msg) : msg("Cannot determine atom type: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct AlternateLocation : public ComplexSpecificException{
  std::string msg;
  AlternateLocation(){
    msg = "Error with alternate locations for atoms";
  }
  AlternateLocation(std::string msg) : msg("Error with alternate locations for atom: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct OccupancyException : public ComplexSpecificException{
  std::string msg;
  OccupancyException(){
    msg = "Cannot process complexes with atoms that have multiple occupancies";
  }
  OccupancyException(std::string msg) : msg("Cannot process complexes with atoms that have multiple occupancies: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct MoleculeSeperationFailure : public ComplexSpecificException{
  std::string msg;
  MoleculeSeperationFailure(){
    msg = "Cannot determine molecule type";
  }
  MoleculeSeperationFailure(std::string msg) : msg("Cannot determine molecule type: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct LackOfTypeException : public ComplexSpecificException{
  std::string msg;
  LackOfTypeException(){
    msg = "Focused type does not exist in current complex";
  }
  LackOfTypeException(std::string msg) : msg("Focused type does not exist in current complex: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct LackOfQueryException : public LackOfTypeException{
  std::string msg;
  LackOfQueryException(){
    msg = "Focused query type does not exist in current complex";
  }
  LackOfQueryException(std::string msg) : msg("Focused query type does not exist in current complex: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct LackOfTargetException : public LackOfTypeException{
  std::string msg;
  LackOfTargetException(){
    msg = "Focused target type does not exist in current complex";
  }
  LackOfTargetException(std::string msg) : msg("Focused target type does not exist in current complex: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct SurfaceClassificationFailure : public ISEException_logic{
  std::string msg;
  SurfaceClassificationFailure(){
    msg = "Particle missing atom type";
  }
  SurfaceClassificationFailure(std::string msg) : msg("Particle missing atom type: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct ParticleListOverflow : public ISEException_runtime{
  std::string msg;
  ParticleListOverflow(){
    msg = "Complex too large";
  }
  ParticleListOverflow(std::string msg) : msg("Complex too large: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct OctreeException : public ISEException_logic{
  std::string msg;
  OctreeException(){
    msg = "ISE sphere octree failure";
  }
  OctreeException(std::string msg) : msg("ISE sphere octree failure: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct ParticleGraphException : public ISEException_logic{
  std::string msg;
  ParticleGraphException(){
    msg = "ISE graph training failure";
  }
  ParticleGraphException(std::string msg) : msg("ISE graph training failure: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct ParticleGraphOverflow : public ParticleGraphException{
  std::string msg;
  ParticleGraphOverflow(){
    msg = "ISE graph overflow";
  }
  ParticleGraphOverflow(std::string msg) : msg("ISE graph overflow: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct BSPException : public ISEException_logic{
  std::string msg;
  BSPException(){
    msg = "Binding Site Prediction failure";
  }
  BSPException(std::string msg) : msg("Binding Site Prediction failure: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct MissingTriangles : public ISEException_runtime{
  std::string msg;
  MissingTriangles(){
    msg = "Cannot predict binding sites without trained triangles";
  }
  MissingTriangles(std::string msg) : msg("Cannot predict binding sites without trained triangles: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};

struct StorageInterfaceConnectionFailure : public ISEException_runtime{
  std::string msg;
  StorageInterfaceConnectionFailure(){
    msg = "Cannot connect to storage interface";
  }
  StorageInterfaceConnectionFailure(std::string msg) : msg("Cannot connect to storage interface: " + msg){}
  virtual const char* what() const throw(){
    return msg.c_str();
  }
};


#endif /* ISEEXCEPTION_HPP */
