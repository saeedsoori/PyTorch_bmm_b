GPU_TARGET =  Turing
FPIC      = -fPIC
CFLAGS    = -O3 $(FPIC) -DADD_ -Wall -MMD

prefix = /usr/local/magma

# --------------------
# directories

# define library directories preferably in your environment, or here.
#ACMLDIR  ?= /opt/acml
CUDADIR  = /usr/local/cuda
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \

INC       = -I$(CUDADIR)/include \
