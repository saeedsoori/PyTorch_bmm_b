GPU_TARGET =  Turing


ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib
FPIC      = -fPIC

prefix = /usr/local/magma

# --------------------
# directories

# define library directories preferably in your environment, or here.
#ACMLDIR  ?= /opt/acml
CUDADIR  = /usr/local/cuda
BLASDIR  = /usr/local/cuda
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \
            -L$(BLASDIR)/

INC       = -I$(CUDADIR)/include \
            -I$(BLAS)/