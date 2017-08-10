OBJS = kernel.o
CFLAGS = -O4 -fPIC

# c compile
COMP = gcc
FILE = kernel.c

# c++ compile
#COMP = g++
#FILE = kernel.cpp

kernel.so: kernel.o
	${COMP} ${CFLAGS} -shared kernel.o -o kernel.so
	rm kernel.o
kernel.o: ${FILE}
	${COMP} ${CFLAGS} ${FILE} -c -o kernel.o
clean:
	rm kernel.so
	rm kernel.o
