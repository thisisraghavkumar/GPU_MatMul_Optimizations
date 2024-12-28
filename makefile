NVCC = /usr/local/cuda-12.4/bin/nvcc
CFLAGS = -std=c++17 -lcublas

SRCDIR = src
KERNELSDIR = $(SRCDIR)/kernels
HELPERSDIR = $(SRCDIR)/helpers
OBJDIR = obj
BINDIR = bin

TARGET=$(BINDIR)/kernel_runner
SOURCES=$(wildcard $(SRCDIR)/*.cu $(KERNELSDIR)/*.cu)
OBJECTS=$(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SOURCES)) \
	$(patsubst $(KERNELSDIR)/%.cu,$(OBJDIR)/%.o,$(SOURCES))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(KERNELSDIR)/mykernels.cuh
	@mkdir -p $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(KERNELSDIR)/%.cu $(KERNELSDIR)/mykernels.cuh
	@mkdir -p $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean