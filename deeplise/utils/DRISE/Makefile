CUDA_INSTALL_PATH := /apps/eb/CUDAcore/11.2.1


CXX := gcc
LINK := nvcc
NVCC := nvcc

# Includes
INCLUDES = -I. -I/usr/local/cuda-11.2.1/include -I./include



# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++14 -gencode=arch=compute_60,code=sm_60 -Iinclude -ljson-c -lxml2 -lfreesasa
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++14 -Iinclude -ljson-c -lxml2 -lfreesasa


LIB_CUDA :=  -L/apps/eb/CUDAcore/11.2.1/lib64 -lcublas -lcudart -lcuda -lgomp -lcusparse -lcusolver -ljson-c -lxml2 -lfreesasa

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))
$(shell mkdir -p ./data)
$(shell mkdir -p ./data/histograms)
$(shell mkdir -p ./data/atomTypes)
$(shell mkdir -p ./data/complexJsons)
$(shell mkdir -p ./data/pdbDatasets)
$(shell mkdir -p ./data/sites)
$(shell mkdir -p ./data/surfaceCheckers)

_OBJS = Molecule.cu.o
_OBJS += bio_maps.cpp.o
_OBJS += Triangle.cu.o
_OBJS += Octree.cu.o
_OBJS += AtomTyping.cpp.o
_OBJS += ParticleList.cu.o
_OBJS += StorageInterface.cpp.o
_OBJS += io_util.cpp.o
_OBJS += cuda_util.cu.o
_OBJS += SurfaceClassifier.cu.o
_OBJS += ParticleGraph.cu.o
_OBJS += ISE.cpp.o
OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

ifeq ($(DEBUG),ON)
	NVCCFLAGS += -DDEBUG
	CXXFLAGS += -DDEBUG
	LINK += -DDEBUG
endif

TARGET = DRISE
LINKLINE = ${LINK} -gencode=arch=compute_60,code=sm_60 ${OBJS} \
 -o ${BINDIR}/${TARGET} ${LIB_CUDA}

.SUFFIXES: .cpp .cu .o

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} ${INCLUDES} -dc $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${BINDIR}/${TARGET}: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f bin/DRISE
	rm -f data/surfaceCheckers/*
	rm -f data/complexJsons/*
	rm -f data/*.ply
	rm -f data/sites/*
	rm -f obj/*
	rm -f data/triangles.json
	rm -f nohup.out
	rm -f data/adjacency/*
	rm -f data/histograms/*
	rm -f data/dna_trainingLog.txt

reset:
	rm -f data/surfaceCheckers/*
	rm -f data/complexJsons/*
	rm -f data/*.ply
	rm -f data/sites/*
	rm -f data/triangles.json
	rm -f data/adjacency/*
	rm -f data/histograms/*
	rm -f data/dna_trainingLog.txt
