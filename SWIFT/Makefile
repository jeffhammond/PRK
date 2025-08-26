# Swift Parallel Research Kernels Makefile
#
# This makefile builds Swift implementations of PRK benchmarks
#

SWIFT := swift
SWIFTC := swiftc

# Compiler flags for optimization
SWIFTFLAGS := -O -whole-module-optimization

# Default target
all: nstream transpose

# Build nstream benchmark
nstream: nstream.swift
	$(SWIFTC) $(SWIFTFLAGS) -o nstream nstream.swift

# Build transpose benchmark  
transpose: transpose.swift
	$(SWIFTC) $(SWIFTFLAGS) -o transpose transpose.swift

# Run tests with default parameters
test: test-nstream test-transpose

test-nstream: nstream
	@echo "Testing nstream..."
	./nstream 10 1000000

test-transpose: transpose
	@echo "Testing transpose..."
	./transpose 10 500

# Run benchmarks with larger parameters
benchmark: benchmark-nstream benchmark-transpose

benchmark-nstream: nstream
	@echo "Benchmarking nstream..."
	./nstream 10 100000000

benchmark-transpose: transpose
	@echo "Benchmarking transpose..."
	./transpose 10 2000

# Clean build artifacts
clean:
	rm -f nstream transpose

# Install Swift (requires Homebrew)
install-swift:
	@echo "Installing Swift via Homebrew..."
	brew install swift

# Check Swift installation
check-swift:
	@echo "Checking Swift installation..."
	@which swift >/dev/null 2>&1 && echo "Swift found at: $$(which swift)" || echo "Swift not found"
	@swift --version 2>/dev/null || echo "Swift version check failed"

# Help target
help:
	@echo "Swift PRK Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build all benchmarks"
	@echo "  nstream       - Build nstream benchmark"
	@echo "  transpose     - Build transpose benchmark"
	@echo "  test          - Run tests with small parameters"
	@echo "  benchmark     - Run benchmarks with larger parameters"
	@echo "  clean         - Remove build artifacts"
	@echo "  install-swift - Install Swift via Homebrew"
	@echo "  check-swift   - Check Swift installation"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all"
	@echo "  make test"
	@echo "  ./nstream 10 1000000"
	@echo "  ./transpose 10 1000"

.PHONY: all test benchmark clean install-swift check-swift help test-nstream test-transpose benchmark-nstream benchmark-transpose
