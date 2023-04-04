
#include "common_includes.h"
#include "Molecule.cuh"
#include "bio_maps.h"
#include "ParticleList.cuh"
#include "SurfaceClassifier.cuh"
#include "ParticleGraph.cuh"
#include "StorageInterface.h"
#include "cuda_util.cuh"
#include "Triangle.cuh"
#include "io_util.h"
#include "Unity.cuh"
#include "Octree.cuh"
#include "freesasa.h"

int main(int argc, char *argv[]){
  try{
    cudaSetDevice(0);
    cuInit(0);
    time_t start = time(nullptr);

    #ifdef DEBUG
      std::cout<<"COMPILED TO DEBUG"<<std::endl;

    #endif


    std::set<MoleculeType> bindConfig;
    int inputType = 1;//default
    std::vector<std::string> input;
    int runConfig = 1;//default
    ConnectorType storageMethod = file_io;//default
    std::string pathToTypingScheme = "./data/atomTypes/typingScheme.csv";//default


    parseArgs(pathToTypingScheme, bindConfig, inputType, input, runConfig, storageMethod, argc, argv);
    int numComplexes = (int) input.size();

    StorageInterface* storageConnector;
    switch(storageMethod){
      case local_database:
        storageConnector = new LocalConnector();
        break;
      case remote_database:
        storageConnector = new RemoteConnector();
        break;
      case file_io:
        storageConnector = new LocalFileConnector();
        break;
    }

    SurfaceInterface surfaceInterface = SurfaceInterface();

    //TODO make these configurable
    fillTypingScheme(pathToTypingScheme);
    surfaceInterface.setFreeSASAParams({FREESASA_LEE_RICHARDS, 1.4, 100, 20, 2});
    surfaceInterface.setDOffset(1.4f);
    float2 queryEdgeConstraints = {2.0f, 20.0f};
    float2 targetEdgeConstraints = {2.0f, 10.0f};
    float interactionThreshold = 6.0f;
    int minInteractions = 6;

    ParticleGraph particleGraph = ParticleGraph(targetEdgeConstraints, queryEdgeConstraints, interactionThreshold, minInteractions);
    particleGraph.setQueryType(bindConfig);
    particleGraph.setTargetType(protein);

    if(runConfig == BSP) particleGraph.setTriangles(storageConnector->readTriangles());
    std::vector<std::string> usedForTraining;
    ParticleList* complex = NULL;
    for(int i = 0; i < numComplexes; ++i){
      start = time(nullptr);
      try{
        std::cout<<"input = "<<input[i]<<std::endl;
        if(complex != NULL){
          delete complex;
          complex = NULL;
        }
        if(storageMethod == local_database || storageMethod == remote_database){
          storageConnector->setProteinId(input[i]);
        }
        complex = (storageMethod == file_io) ? new ParticleList(input[i]) : storageConnector->readParticles();
        if(runConfig != SURFACE){
          complex->classifyAtoms(classifyAtomsISEResidue);
          #ifdef DEBUG
            writeAtomTypeChecker(complex->molecules, complex->identifier);
          #endif
          if(!complex->checkForTypingContinuity(protein)) throw LackOfTypeException("protein");
        }
        if(inputType == file_io) storageConnector->setProteinId(complex->identifier);
        if(runConfig != ATOMTYPE){
          surfaceInterface.extractFreeSASA(complex->getMolecules(protein));
          #ifdef DEBUG
            if(complex->fileType == PDB) writeSurfaceChecker(complex->molecules, complex->identifier);
          #endif
          if(runConfig != SURFACE && runConfig != ATOMTYPE){
            if(runConfig == TRAIN){
              particleGraph.determineBindingSiteTruth(complex);
              particleGraph.buildParticleGraph(complex);
              usedForTraining.push_back(complex->identifier);
            }
            else if(runConfig == BSP){
              particleGraph.updateScores(complex);
              complex->determineResidueScores();
              prepareVisualizationFile(complex->molecules, complex->identifier, protein, false);
            }
          }
        }
        cudaDeviceSynchronize();//may be unnessary
        storageConnector->setCurrentComplex(complex);
        storageConnector->writeParticles();

        if(runConfig == TRAIN){
          std::cout << "time elapsed = " << difftime(time(nullptr), start)
          <<" seconds for "<<complex->identifier<< ".("<<usedForTraining.size()
          <<"/"<<numComplexes - (i - usedForTraining.size()) - 1<<")"<<std::endl;
        }
        else{
          std::cout << "time elapsed = " << difftime(time(nullptr), start)
          <<" seconds for "<<complex->identifier<< ".("<<i+1<<"/"<<numComplexes<<")"<<std::endl;
        }

      }
      catch (const ComplexSpecificException  &e){
        std::cerr << "Caught exception- " << e.what() << '\n';
      }
      std::cout<<"---------------------------------------------------\n\n"
      <<"---------------------------------------------------"<<std::endl;
    }
    storageConnector->setRadii(radius);
    storageConnector->setAtomTypes(atomTypeMap);
    //binding site prediction using existing triangle scores
    if(runConfig == TRAIN){
      particleGraph.normalizeTriangles();
      std::vector<Triangle> triangles = particleGraph.getTriangles();
      //NOTE this is just sorting for DRISE
      storageConnector->setTriangles(triangles);
      storageConnector->writeTriangles();
    }

    delete storageConnector;
    return 0;
  }
  catch (const ISEException &e){
    std::cerr << "Caught exception: " << e.what() << '\n';
    std::exit(1);
  }
  catch (const std::exception &e){
    std::cerr << "Caught exception: " << e.what() << '\n';
    std::exit(1);
  }
  catch (...){
    std::cerr << "Caught unknown exception\n";
    std::exit(1);
  }
}
