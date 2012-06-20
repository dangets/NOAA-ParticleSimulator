
ParticleSimulation: ParticleSimulation.cu loadShaders.cpp
	nvcc -o $@ $^ -lGLEW -lglfw -ljsoncpp

clean:
	rm -f *.o
	rm -f ParticleSimulation
