CC        = clang
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = cpu
LAUNCHER  =


# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c11 -Wall

# Linker flags
LDFLAGS = -lm

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=
endif

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g  
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O0
endif


./srad:	./main.o \
	./util/graphics/graphics.o \
	./util/graphics/resize.o \
	./util/timer/timer.o
	$(CC) $(CFLAGS)	./main.o \
		./util/graphics/graphics.o \
		./util/graphics/resize.o \
		./util/timer/timer.o \
		-lm \
		$(LDFLAGS) -o srad

./main.o: ./main.c ./main.h
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./main.c \
		-c \
		-o ./main.o

./util/graphics/graphics.o:	./util/graphics/graphics.h \
	./util/graphics/graphics.c
	$(CC) $(CFLAGS) ./util/graphics/graphics.c \
		-c \
		-o ./util/graphics/graphics.o

./util/graphics/resize.o:	./util/graphics/resize.h \
	./util/graphics/resize.c
	$(CC) $(CFLAGS) ./util/graphics/resize.c \
		-c \
		-o ./util/graphics/resize.o

./util/timer/timer.o:	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC) $(CFLAGS) ./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o

clean:
	rm -f ./*.o srad ./image_out.pgm ./util/graphics/*.o ./util/timer/*.o

run: srad
	$(LAUNCHER) ./srad 1000 0.5 502 458
