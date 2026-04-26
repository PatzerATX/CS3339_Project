#===============================================================================
# User Options — Optimized for ARM64 Apple Silicon at -O0
#===============================================================================

CC       = icpx
OPTIMIZE = yes
DEBUG    = no
DEVICE   = gpu
LAUNCHER =

# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O0
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS += -qopenmp
endif

PREFIX = benchmark_Claude_optimized

all: kmeans

kmeans: $(PREFIX)_cluster.o $(PREFIX)_getopt.o $(PREFIX)_read_input.o $(PREFIX)_rmse.o $(PREFIX)_kmeans.c
	$(CC) $(KERNEL_DIM) $(CFLAGS) $^ -o kmeans $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $< -c

$(PREFIX)_kmeans.o:      $(PREFIX)_kmeans.c      $(PREFIX)_kmeans.h
$(PREFIX)_cluster.o:     $(PREFIX)_cluster.c     $(PREFIX)_kmeans.h
$(PREFIX)_getopt.o:      $(PREFIX)_getopt.c      $(PREFIX)_getopt.h
$(PREFIX)_read_input.o:  $(PREFIX)_read_input.c  $(PREFIX)_kmeans.h
$(PREFIX)_rmse.o:        $(PREFIX)_rmse.c        $(PREFIX)_kmeans.h

clean:
	rm -f *.o *~ kmeans *.linkinfo

run: kmeans
	$(LAUNCHER) ./kmeans -r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup
