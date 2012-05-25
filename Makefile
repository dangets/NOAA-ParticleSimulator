
all: readINL testINLReader

clean:
	rm -f *.o
	rm -f readINL
	rm -f testINLReader

readINL: readINL.cpp

testINLReader: testINLReader.cpp INLReader.o
