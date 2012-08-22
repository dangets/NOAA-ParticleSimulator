.DEFAULT: main


CC  = nvcc
CXX = nvcc

OPENGL_LIBS = -lglfw -lGLEW
VTK_INCL = -I/usr/include/vtk-5.4/
VTK_LIBS = -lvtkCommon -lvtkIO
OTHER_INCL = -I./include
#OTHER_LIBS  = -L./lib -ljsoncpp
OTHER_FILES = ./lib/libjsoncpp.a

NVCC_INCL = $(VTK_INCL) $(OTHER_INCL)
NVCC_LIBS = $(OPENGL_LIBS) $(VTK_LIBS) $(OTHER_LIBS)

CUDA_SRCS = \
	    ParticleSet.cu \
	    ParticleSetOpenGLVBO.cu \
	    WindData.cu \
	    advect_runge_kutta.cu \
	    vtk_io.cu \
	    # \
	    advect_original.cu \


CUDA_OBJS = $(addsuffix .o, $(basename $(CUDA_SRCS)))


%.o : %.cu
	nvcc -c $^ $(NVCC_INCL)

% : %.cu
	nvcc -o $@  $^ $(NVCC_INCL)


# NOTE: for some reason 'main.cu' has to be at the END of the files to be compiled
#	(nvcc bug as of v4.0)
main: loadShaders.o $(CUDA_OBJS) main.cu
	nvcc -o $@ $^ $(OTHER_FILES) $(NVCC_INCL) $(NVCC_LIBS)

testWindData: WindData.o testWindData.cu
	nvcc -o $@ $^ $(NVCC_INCL) $(NVCC_LIBS)


.PHONY : clean
clean:
	rm -f *.o
	rm -f main
