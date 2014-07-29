# Compiler Options
GCC=nvcc
STD=-Xcompiler "-Ox"
OPT=-O3
ARCH=-arch=sm_30

# Directories
BIN=bin
LIB=lib
SRC=src

# Compiler command
COMP=$(GCC) $(STD) $(OPT) $(ARCH) 

all: Refactor

Refactor : $(SRC)/Refactor_v1.cu
	$(COMP) $(SRC)/Refactor_v1.cu -o $(BIN)/Refactor.exe
	