CC          = icpx
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu
LAUNCHER    =

PREFIX = benchmark_Claude_optimized

# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif

# Optimization Flags — set to -O0 per benchmark requirement
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O0
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS += -qopenmp
endif

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort: $(PREFIX)_hybridsort.o $(PREFIX)_bucketsort.o $(PREFIX)_mergesort.o
	$(CC) $(CFLAGS) -o hybridsort $(VERIFY) $(OUTPUT) $(TIMER) \
		$(PREFIX)_hybridsort.o \
		$(PREFIX)_bucketsort.o \
		$(PREFIX)_mergesort.o \
		$(LDFLAGS)

$(PREFIX)_hybridsort.o: $(PREFIX)_hybridsort.c \
	$(PREFIX)_bucketsort.h $(PREFIX)_mergesort.h
	$(CC) $(CFLAGS) -o $(PREFIX)_hybridsort.o -c $(PREFIX)_hybridsort.c

$(PREFIX)_bucketsort.o: $(PREFIX)_bucketsort.c \
	$(PREFIX)_kernel_bucketprefix.h \
	$(PREFIX)_kernel_histogram.h \
	$(PREFIX)_kernel_bucketcount.h \
	$(PREFIX)_kernel_bucketsort.h
	$(CC) $(CFLAGS) -o $(PREFIX)_bucketsort.o -c $(PREFIX)_bucketsort.c

$(PREFIX)_mergesort.o: $(PREFIX)_mergesort.c \
	$(PREFIX)_mergesort.h \
	$(PREFIX)_kernel_mergeSortPass.h
	$(CC) $(CFLAGS) -o $(PREFIX)_mergesort.o -c $(PREFIX)_mergesort.c

run: hybridsort
	$(LAUNCHER) ./hybridsort r

clean:
	rm -f *.o hybridsort
