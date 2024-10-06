CXX=g++ -m64 -march=native
CXXFLAGS=-I../common -Iobjs/ -O3 -Wall
ISPC=ispc
# note: requires AVX2 capable machine
ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64 --pic

APP_NAME=sqrt
OBJDIR=objs
COMMONDIR=../common

PPM_CXX=$(COMMONDIR)/ppm.cpp
PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
	/bin/mkdir -p $(OBJDIR)/

clean:
	/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

# Updated OBJS to include sqrt_avx2.o
OBJS=$(OBJDIR)/main.o $(OBJDIR)/sqrtSerial.o $(OBJDIR)/sqrt_ispc.o $(OBJDIR)/sqrt_avx2.o $(PPM_OBJ) $(TASKSYS_OBJ)

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm $(TASKSYS_LIB)

# Generic rule for compiling .cpp files in the current directory
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Generic rule for compiling .cpp files in the COMMONDIR
$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Specific rule for compiling sqrt_avx2.o
$(OBJDIR)/sqrt_avx2.o: sqrt_avx2.cpp sqrt_avx2.h
	$(CXX) $(CXXFLAGS) -c sqrt_avx2.cpp -o $(OBJDIR)/sqrt_avx2.o

# ISPC related rules
$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h
