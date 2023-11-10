all:	houghBase

houghBase:	houghBase.cu pgm.o
	nvcc -arch=sm_20 houghBase.cu pgm.o -o houghBase

pgm.o:	./common/pgm.cpp
	g++ -c ./common/pgm.cpp -o ./pgm.o
