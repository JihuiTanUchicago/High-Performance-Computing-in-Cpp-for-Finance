# Compiler
CXX = clang++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -march=native  # Optimization and warnings

# Executable name
TARGET = pricer_exec

# Source files
SRCS = european_call_black_scholes_pricer.cpp PricingData.cpp

# Object files (replace .cpp with .o)
OBJS = $(SRCS:.cpp=.o)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET) $(OBJS)
