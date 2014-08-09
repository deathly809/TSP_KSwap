
# Compiler Options
GCC=nvcc
CMP=-Xcompiler "-Ox"
OPT=-O3 -use_fast_math -lineinfo --ptxas-options=-v
ARCH=-arch=sm_30

# Directories
BIN=bin
LIB=lib
SRC=src

# Compiler command
COMP=$(GCC) $(CMP) $(OPT) $(ARCH) 

all: $(BIN)/Refactor_v1.exe $(BIN)/Refactor_v2.exe $(BIN)/Refactor_v3.exe $(BIN)/Refactor_v3a.exe $(BIN)/Refactor_v4.exe $(BIN)/KSwaps.exe $(BIN)/TSP_GPU21.exe

$(BIN)/Refactor_v1.exe : $(SRC)/Refactor_v1.cu
	$(COMP) $(SRC)/Refactor_v1.cu -o $(BIN)/Refactor_v1.exe
	@echo.
	
$(BIN)/Refactor_v2.exe : $(SRC)/Refactor_v2.cu
	$(COMP) $(SRC)/Refactor_v2.cu -o $(BIN)/Refactor_v2.exe
	@echo.
	
$(BIN)/Refactor_v3.exe : $(SRC)/Refactor_v3.cu
	$(COMP) $(SRC)/Refactor_v3.cu -o $(BIN)/Refactor_v3.exe
	@echo.
	
$(BIN)/Refactor_v3a.exe : $(SRC)/Refactor_v3a.cu
	$(COMP) $(SRC)/Refactor_v3a.cu -o $(BIN)/Refactor_v3a.exe
	
$(BIN)/Refactor_v4.exe : $(SRC)/Refactor_v4.cu
	$(COMP) $(SRC)/Refactor_v4.cu -o $(BIN)/Refactor_v4.exe
	@echo.
	
$(BIN)/KSwaps.exe: $(SRC)/Final/TwoOptKSwaps.cu
	$(COMP) $(SRC)/Final/TwoOptKSwaps.cu -o $(BIN)/KSwaps.exe
	@echo.
	
$(BIN)/TSP_GPU21.exe: $(SRC)/Final/TSP_GPU21.cu
	$(COMP) $(SRC)/Final/TSP_GPU21.cu -o $(BIN)/TSP_GPU21.exe
	@echo.
	
clean:
	rm bin/*