GPU_TARGET =  Turing
CC        = gcc
CXX       = g++
NVCC      = nvcc
#FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

#FPIC      = -fPIC
CFLAGS    = -O3 $(FPIC) -fopenmp -DADD_ -Wall -Wno-strict-aliasing -Wshadow 
CXXFLAGS := $(CFLAGS) -std=c++11


# --------------------
# directories

# define library directories preferably in your environment, or here.
#ACMLDIR  ?= /opt/acml
CUDADIR  = /usr/local/cuda
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 

INC       = -I$(CUDADIR)/include