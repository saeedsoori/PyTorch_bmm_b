GPU_TARGET =  Turing
FPIC      = -fPIC
CFLAGS    = -O3 $(FPIC) -DADD_ -Wall -MMD
CFLAGS    = -O3 $(FPIC) -DADD_ -Wall -MMD
CXXFLAGS  = $(CFLAGS) -std=c++11
NVCCFLAGS = -O3         -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function" -std=c++11
FFLAGS    = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS  = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
prefix = /usr/local/magma

# --------------------
# libraries

# see MKL Link Advisor at http://software.intel.com/sites/products/mkl/
# gcc/gfortran with MKL 10.3, GNU OpenMP threads (use -fopenmp in CFLAGS, LDFLAGS)
LIB       = -lpthread -lstdc++ -lm -lgfortran

# Supposedly, gcc can use Intel threads (libiomp5) instead, but be careful that
# libiomp5 and libgomp are NOT BOTH linked. Above, we use gnu threads as a safer option.
# gcc/gfortran with MKL 10.3, Intel OpenMP threads (remove -fopenmp from LDFLAGS above)
#LIB       = -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -lgfortran

LIB      += -lcublas -lcusparse -lcudart -lcudadevrt

# directories

# define library directories preferably in your environment, or here.
#ACMLDIR  ?= /opt/acml
CUDADIR  = /usr/local/cuda
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \

INC       = -I$(CUDADIR)/include \
