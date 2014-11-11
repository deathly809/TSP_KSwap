
# Compiler Options
GCC=nvcc
CMP=-Xcompiler "-Ox"
OPT=-O3 -use_fast_math -lineinfo --ptxas-options=-v -m64
ARCH2=-gencode arch=compute_20,code=sm_20
ARCH3=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# Directories
BIN=bin
LIB=lib
SRC=src

# Compiler command
COMP=$(GCC) $(CMP) $(OPT) $(ARCH3)

all:	$(BIN)/Refactor_v0.exe $(BIN)/Refactor_v1.exe $(BIN)/Refactor_v2.exe $(BIN)/Refactor_v3.exe $(BIN)/Refactor_v3a.exe $(BIN)/Refactor_v4.exe \
		$(BIN)/Refactor_v4a.exe $(BIN)/Refactor_v4b.exe $(BIN)/Refactor_v4c.exe $(BIN)/KSwaps.exe $(BIN)/TSP_GPU21.exe \
		$(BIN)/Refactor_v5.exe

$(BIN)/Refactor_v0.exe : $(SRC)/Refactor_v0.cu
	$(COMP) $(SRC)/Refactor_v0.cu -o $(BIN)/Refactor_v0.exe
	@echo .

$(BIN)/Refactor_v1.exe : $(SRC)/Refactor_v1.cu
	$(COMP) $(SRC)/Refactor_v1.cu -o $(BIN)/Refactor_v1.exe
	@echo .
	
$(BIN)/Refactor_v2.exe : $(SRC)/Refactor_v2.cu
	$(COMP) $(SRC)/Refactor_v2.cu -o $(BIN)/Refactor_v2.exe
	@echo .
	
$(BIN)/Refactor_v3.exe : $(SRC)/Refactor_v3.cu
	$(COMP) $(SRC)/Refactor_v3.cu -o $(BIN)/Refactor_v3.exe
	@echo .
	
$(BIN)/Refactor_v3a.exe : $(SRC)/Refactor_v3a.cu
	$(COMP) $(SRC)/Refactor_v3a.cu -o $(BIN)/Refactor_v3a.exe
	@echo .
	
$(BIN)/Refactor_v4.exe : $(SRC)/Refactor_v4.cu
	$(COMP) $(SRC)/Refactor_v4.cu -o $(BIN)/Refactor_v4.exe
	@echo .
	
$(BIN)/Refactor_v4a.exe : $(SRC)/Refactor_v4a.cu
	$(COMP) $(SRC)/Refactor_v4a.cu -o $(BIN)/Refactor_v4a.exe
	@echo .
	
$(BIN)/Refactor_v4b.exe : $(SRC)/Refactor_v4b.cu
	$(COMP) $(SRC)/Refactor_v4b.cu -o $(BIN)/Refactor_v4b.exe
	@echo .

$(BIN)/Refactor_v4c.exe : $(SRC)/Refactor_v4c.cu
	$(COMP) $(SRC)/Refactor_v4c.cu -o $(BIN)/Refactor_v4c.exe
	@echoi .

$(BIN)/Refactor_v5.exe : $(SRC)/Refactor_v5.cu
	$(COMP) $(SRC)/Refactor_v5.cu -o $(BIN)/Refactor_v5.exe
	@echo .


	
$(BIN)/KSwaps.exe: $(SRC)/Final/TwoOptKSwaps.cu
	$(COMP) $(SRC)/Final/TwoOptKSwaps.cu -o $(BIN)/KSwaps.exe
	@echo .
	
$(BIN)/TSP_GPU21.exe: $(SRC)/Final/TSP_GPU21.cu
	$(COMP) $(SRC)/Final/TSP_GPU21.cu -o $(BIN)/TSP_GPU21.exe
	@echo .
	
cache:
	$(COMP) $(SRC)/Cache.cu -o $(BIN)/Cache.exe
	
clean:
	rm bin/*
