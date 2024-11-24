build: ./src/helper.cu ./src/helper.cuh ./src/mnist.hpp ./src/main.cu Makefile
	mkdir -p out
	nvcc ./src/main.cu -o ./out/main

run:
	./out/main

all: clean build run

clean:
	rm -rf ./out
