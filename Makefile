CXX      ?= clang++
CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra -Werror -Isrc
BIN := bin
SRC := src
.PHONY: all test clean
all: $(BIN)/bench
$(BIN)/bench: $(SRC)/bench.cpp $(SRC)/gelu.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(SRC)/bench.cpp -o $@
$(BIN)/test_gelu: tests/test_gelu.cpp $(SRC)/gelu.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) tests/test_gelu.cpp -o $@
test: $(BIN)/test_gelu
	$(BIN)/test_gelu
$(BIN):
	mkdir -p $(BIN)
clean:
	rm -rf $(BIN)
