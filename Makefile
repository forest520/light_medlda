SHELL   =  /bin/bash
PROJECT := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# https://github.com/xunzheng/third_party.git
THIRD_PARTY     = $(PROJECT)/third_party
THIRD_PARTY_LIB = $(THIRD_PARTY)/lib
THIRD_PARTY_INC = $(THIRD_PARTY)/include

CXX      = g++
CXXFLAGS = -O3 \
           -std=c++11 \
           -Wall \
           -Wno-unused-function \
	   -fno-builtin-malloc \
           -fno-builtin-calloc \
           -fno-builtin-realloc \
	   -fno-builtin-free \
           -I$(THIRD_PARTY_INC)
LDFLAGS  = -Wl,-rpath,$(THIRD_PARTY_LIB) \
           -L$(THIRD_PARTY_LIB)

BIN = med
SRC = $(wildcard *.cc)
HDR = $(wildcard *.h)
DYN = $(LDFLAGS) -lm -lrt -lgflags -lglog -ltcmalloc

all: $(BIN)

$(BIN): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) $(DYN) -o $@

clean:
	rm -f $(BIN)

.PHONY: all clean
