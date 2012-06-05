
all: readINL testINLReader

clean:
	rm -f *.o
	rm -f readINL
	rm -f testINLReader
	rm -f testParticleSource

readINL: readINL.cpp

testINLReader: testINLReader.cpp INLReader.o

testParticleSource: testParticleSource.cpp ParticleSource.hpp
