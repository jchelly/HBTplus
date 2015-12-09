SRC_COMM=$(wildcard src/*.cpp) $(wildcard src/io/*.cpp)
OBJS_COMM=$(SRC_COMM:%.cpp=%.o)

SRC=$(wildcard *.cpp)
EXE=HBT HBTdouble

default: HBT
include Makefile.inc

$(EXE): $(OBJS_COMM)

HBTdouble: CXXFLAGS+=-DHBT_REAL8 -DHBT_INT8 
HBTdouble: HBT.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

depend:
	makedepend --$(CXXFLAGS)-- -Y $(SRC)
	
synccosma: clean
	rsync -avzL $(shell pwd)/ jvbq85@cosma-a:data/HBT2/omp
	
synccosmalocal: clean
	rsync -avzL -e "ssh -p 4800" $(shell pwd)/ jvbq85@localhost:data/HBT2/omp
