CC          = clang++
OPTIMIZE    = no
DEBUG       = no
DEVICE      = cpu
LAUNCHER    =

# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif

# Optimization level: explicitly O0 per benchmark target
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
else
  CFLAGS += -O0
endif

# CPU OpenMP via clang on Apple Silicon (ARM64).
# The original GPU path used -fiopenmp -fopenmp-targets=spir64 (Intel SPIR-V).
# On Apple Silicon there is no SPIR-V GPU; using plain -fopenmp routes all
# omp target regions to execute on the host CPU without GPU-offload overhead.
CFLAGS += -fopenmp

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort: hybridsort_ClaudeSonnet_optimized.o bucketsort_ClaudeSonnet_optimized.o mergesort_ClaudeSonnet_optimized.o
	$(CC) $(CFLAGS) -o hybridsort $(VERIFY) $(OUTPUT) $(TIMER) \
		hybridsort_ClaudeSonnet_optimized.o \
		bucketsort_ClaudeSonnet_optimized.o \
		mergesort_ClaudeSonnet_optimized.o \
		$(LDFLAGS)

hybridsort_ClaudeSonnet_optimized.o: hybridsort_ClaudeSonnet_optimized.c bucketsort.h mergesort.h
	$(CC) $(CFLAGS) -DTIMER -o hybridsort_ClaudeSonnet_optimized.o -c hybridsort_ClaudeSonnet_optimized.c

bucketsort_ClaudeSonnet_optimized.o: bucketsort_ClaudeSonnet_optimized.c \
	kernel_bucketprefix.h \
	kernel_histogram.h \
	kernel_bucketcount.h \
	kernel_bucketsort.h
	$(CC) $(CFLAGS) -o bucketsort_ClaudeSonnet_optimized.o -c bucketsort_ClaudeSonnet_optimized.c

mergesort_ClaudeSonnet_optimized.o: mergesort_ClaudeSonnet_optimized.c mergesort.h kernel_mergeSortPass.h
	$(CC) $(CFLAGS) -o mergesort_ClaudeSonnet_optimized.o -c mergesort_ClaudeSonnet_optimized.c

run: hybridsort
	$(LAUNCHER) ./hybridsort r

clean:
	rm -f *.o hybridsort
