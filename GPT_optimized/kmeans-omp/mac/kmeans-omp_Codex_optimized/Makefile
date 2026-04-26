CXX ?= clang++
OPT_LEVEL ?= -O0
EXTRA_CFLAGS ?=

TARGET = benchmark_Codex_optimized_kmeans
SOURCES = \
	benchmark_Codex_optimized_cluster.c \
	benchmark_Codex_optimized_read_input.c \
	benchmark_Codex_optimized_rmse.c \
	benchmark_Codex_optimized_kmeans.c
OBJECTS = $(SOURCES:.c=.o)

CXXFLAGS := $(EXTRA_CFLAGS) $(OPT_LEVEL) -std=c++17 -Wall
LDFLAGS := -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) *~ *.linkinfo
