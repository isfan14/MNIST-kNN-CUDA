build:
	mkdir -p out
	nvcc ./src/main.cu -o ./out/main

run:
	./out/main

all: clean build run

clean:
	rm -rf ./out
