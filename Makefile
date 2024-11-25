SOURCES := $(wildcard src/*/main.cu)
GLOBAL_CPP_UTILS := $(wildcard src/utils/*.cpp)
GLOBAL_CUDA_UTILS := $(wildcard src/utils/*.cu)
OUTPUTS := $(SOURCES:src/%/main.cu=out/%.o)

all: out logs $(OUTPUTS)

out logs:
	mkdir -p $@

clean:
	rm -f out/*

.SECONDEXPANSION:

LOCAL_CPP_UTILS := $$(wildcard src/%/utils/*.cpp)
LOCAL_CUDA_UTILS := $(wildcard src/%/utils/*.cu)

$(OUTPUTS): out/%.o: $(GLOBAL_CPP_UTILS) $(GLOBAL_CUDA_UTILS) $(LOCAL_CPP_UTILS) $(LOCAL_CUDA_UTILS) src/%/main.cu
	nvcc $^ -o $@

