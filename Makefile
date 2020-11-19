CC := @g++
CUCC := nvcc
ECHO := @echo
SRCDIR := src
OBJDIR := objs
BINDIR := ./build

# -gencode=arch=compute_60,code=sm_60
OUTNAME := gpu_demo
# -m64 -fPIC -g -fopenmp -w -O3 -pthread
# -m64 -Xcompiler -fPIC -g -w -O3 -gencode=arch=compute_75,code=sm_75
# -gencode 参见： https://www.cnblogs.com/phillee/p/12049208.html
CFLAGS := -std=c++11 -g -fPIC -m64 -fopenmp -w -O3 -pthread
CUFLAGS := -std=c++11 -g -m64 -Xcompiler -w -O3
INC_CUDA := /usr/local/cuda/include 
INCS := $(INC_CUDA)
INCS := $(foreach inc, $(INCS), -I$(inc))

LIB_CUDA := /usr/local/cuda/lib64
LIBS := $(LIB_CUDA)
RPATH := $(foreach lib, $(LIBS),-Wl,-rpath=$(lib))
LIBS := $(foreach lib, $(LIBS),-L$(lib))

LD_CUDA := cuda cudart curand
LD_SYS := dl stdc++ pthread
LDS := $(LD_CUDA) $(LD_SYS)
LDS := $(foreach lib, $(LDS), -l$(lib))

SRCS := $(shell cd $(SRCDIR) && find -name "*.cpp")
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
OBJS := $(foreach item,$(OBJS),$(OBJDIR)/$(item))
CUS := $(shell cd $(SRCDIR) && find -name "*.cu")
CUOBJS := $(patsubst %.cu,%.o,$(CUS))
CUOBJS := $(foreach item,$(CUOBJS),$(OBJDIR)/$(item))
CS := $(shell cd $(SRCDIR) && find -name "*.c")
COBJS := $(patsubst %.c,%.o,$(CS))
COBJS := $(foreach item,$(COBJS),$(OBJDIR)/$(item))
OBJS := $(subst /./,/,$(OBJS))
CUOBJS := $(subst /./,/,$(CUOBJS))
COBJS := $(subst /./,/,$(COBJS))

all : $(BINDIR)/$(OUTNAME)
	$(ECHO) Compile done.

run: all
	@cd $(BINDIR) && ./$(OUTNAME);

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS) $(COBJS)
	$(ECHO) Linking: $@
	@if [ ! -d $(BINDIR) ]; then mkdir $(BINDIR); fi
	@$(CC) $(CFLAGS) $(LIBS) -o $@ $^ $(LDS) $(RPATH)

$(CUOBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CUCC) $(CUFLAGS) $(INCS) -c -o $@ $<

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

$(COBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)/$(OUTNAME)