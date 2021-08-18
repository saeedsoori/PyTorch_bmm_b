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
# directories

# define library directories preferably in your environment, or here.
#ACMLDIR  ?= /opt/acml
CUDADIR  = /usr/local/cuda
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \

INC       = -I$(CUDADIR)/include \
