# Makefile for Linear Regression C++/Python Project

# Configuration
MODULE_NAME   := linear_regression
WRAPPER_SRC   := linear_regression_wrapper.cpp
EXT_SUFFIX    := $(shell python3-config --extension-suffix)
CXX           := g++
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)

# Compiler flags
CXXFLAGS      := -O3 -Wall -shared -std=c++17 -fPIC
LDFLAGS       := -Wl,-rpath,$(CONDA_PREFIX)/lib

# Target
TARGET        := $(MODULE_NAME)$(EXT_SUFFIX)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(WRAPPER_SRC) linear_regression.hpp
	@echo "Compiling C++ extension with Conda's libraries..."
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $(WRAPPER_SRC) -o $(TARGET) $(LDFLAGS)
	@echo "Verifying library links..."
	@ldd $(TARGET) | grep libstdc++ || true

clean:
	@rm -fv $(MODULE_NAME)*.so

linear_regression: $(TARGET)
	@echo "Running Python Interface..."
	python linear_regression.py

check-env:
ifndef CONDA_PREFIX
	$(error Conda environment '$(CONDA_ENV)' is not activated!)
endif