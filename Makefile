CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -Iinclude
LDFLAGS = -lm

SRC = src/wasserstein_ot.c
OBJ = $(SRC:.c=.o)

.PHONY: all test clean

all: libwasserstein_ot.a test_wasserstein

src/wasserstein_ot.o: src/wasserstein_ot.c include/wasserstein_ot.h
	$(CC) $(CFLAGS) -c -o $@ $<

libwasserstein_ot.a: src/wasserstein_ot.o
	ar rcs $@ $<

test_wasserstein: tests/test_wasserstein.c libwasserstein_ot.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lwasserstein_ot $(LDFLAGS)

test: test_wasserstein
	./test_wasserstein

clean:
	rm -f src/*.o libwasserstein_ot.a test_wasserstein
