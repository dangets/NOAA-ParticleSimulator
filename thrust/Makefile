
ParticleSimulation: ParticleSimulation.cu loadShaders.cpp
	nvcc -o $@ $^ -lGLEW -lglfw

clean:
	rm -f *.o
	rm -f ParticleSimulation
