#ifndef STORAGEINTERFACE_H
#define STORAGEINTERFACE_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "common_includes.h"
#include "ParticleList.cuh"
#include "io_util.h"

//TODO change pdb to complex

class StorageInterface{

  public:

    ParticleList* complex;
    void setCurrentComplex(ParticleList* currentComplex){
      this->complex = currentComplex;
      this->id = currentComplex->identifier;
    }

    std::string id;
    void setProteinId(std::string id){
      this->id = id;
    }

    std::vector<Triangle> triangles;
    void setTriangles(std::vector<Triangle> triangles){
      this->triangles = triangles;
    }

    std::map<std::string, float> radii;
    void setRadii(std::map<std::string, float> radii){
      this->radii = radii;
    }
    std::map<int, char> moleculeTypes;
    void setMoleculeTypes(std::map<int, char> moleculeTypes){
      this->moleculeTypes = moleculeTypes;
    }
    std::map<std::string, unsigned char> atomTypes;
    void setAtomTypes(std::map<std::string, unsigned char> atomTypes){
      this->atomTypes = atomTypes;
    }


    //API
    virtual void writeParticles() =0;
    virtual ParticleList* readParticles() =0;
    virtual void writeTriangles() =0;
    virtual std::vector<Triangle> readTriangles() =0;
    virtual ~StorageInterface(){}

};

//This implementation connects to a remote DB server via HTTP
class RemoteConnector : public StorageInterface{

  public:

    int portno;
    std::string host;

    struct hostent *server;
    struct sockaddr_in serv_addr;
    int sockfd, bytes, sent, received, total, message_size;
    char *message, response[4096];

    RemoteConnector(); //constructor
    ~RemoteConnector();

    void writeParticles();
    ParticleList* readParticles();
    void writeTriangles();
    std::vector<Triangle> readTriangles();

};

//This implementation connects to a DB server on the local machine
class LocalConnector : public StorageInterface{

  public:
    LocalConnector();
    ~LocalConnector();

    void writeParticles();
    ParticleList* readParticles();
    void writeTriangles();
    std::vector<Triangle> readTriangles();

};

//DEPRECATED. This implementation uses file io to store and read data
//def slower than local db, probably faster than remote unless you have a dedicated ethernet link
class LocalFileConnector : public StorageInterface{

public:
  LocalFileConnector();
  ~LocalFileConnector();

  void writeParticles();
  ParticleList* readParticles();
  void writeTriangles();
  std::vector<Triangle> readTriangles();

};

#endif /* STORAGEINTERFACE_H */
