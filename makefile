NVCC = /usr/local/cuda-12.4/bin/nvcc
CFLAGS = -std=c++17 -lcublas -g -Xptxas=-v -lnvToolsExt

SRCDIR = src
KERNELSDIR = $(SRCDIR)/kernels
HELPERSDIR = $(SRCDIR)/helpers
OBJDIR = obj
BINDIR = bin

TARGET=$(BINDIR)/kernel_runner
SOURCES=$(wildcard $(SRCDIR)/*.cu $(KERNELSDIR)/*.cu)
OBJECTS=$(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SOURCES)) 

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(KERNELSDIR)/mykernels.cuh $(HELPERSDIR)/myhelpers.h
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OBJDIR)/kernels
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean
