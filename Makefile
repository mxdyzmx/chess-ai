CXX := g++
CXXFLAGS := -O3 -march=native -flto -pthread -std=c++17
WARN := -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter
LDFLAGS := -flto -pthread

SRCDIR := src
OBJDIR := obj
TARGET := chessai

# Source files
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

.PHONY: all clean release debug profile

all: release

# Release build (default)
release: CXXFLAGS += $(WARN)
release: $(TARGET)

# Debug build (slow, no optimizations)
debug: CXXFLAGS = -g -O0 -DDEBUG -pthread -std=c++17 $(WARN)
debug: $(TARGET)

# Profile build (for performance analysis)
profile: CXXFLAGS = -O3 -march=native -g -fprofile-generate -pthread -std=c++17 $(WARN)
profile: LDFLAGS = -fprofile-generate -pthread
profile: $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^
	@echo ""
	@echo "Build complete: $(TARGET)"
	@echo "Run with: ./$(TARGET)"
	@echo "Benchmark: ./$(TARGET) --bench"

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)
	@echo "Cleaned."

# Install (copy to /usr/local/bin)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	@echo "Installed to /usr/local/bin/$(TARGET)"

# Benchmark target
bench: $(TARGET)
	./$(TARGET) --bench

# Dependency generation
$(OBJDIR)/%.d: $(SRCDIR)/%.cpp | $(OBJDIR)
	@$(CXX) -MM -MT $(@:.d=.o) $(CXXFLAGS) $< -o $@

-include $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.d)
