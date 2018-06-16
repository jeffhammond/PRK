SWIFT 	= swiftc
SWFLAGS	= -target x86_64-apple-macosx10.11

all: vector

vector: vector.swift
	$(SWIFT) $(SWFLAGS) $< -o $@

clean:
	-rm -f vector
