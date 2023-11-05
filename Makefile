SRC=$(wildcard *.c)
LIBS=-lm
CC_ARM=/home/fedor/NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang 
CC_INTEL=cc

all: arm64 intel

intel: $(SRC)
		$(CC_INTEL) -c $^ $(LIBS)
		ar r tinyspeex_intel.a *.o

arm64: $(SRC)
		$(CC_ARM) -c $^ $(LIBS)
		ar r tinyspeex_arm64.a *.o

clean:
		rm *.a *.o
