SWIFT 	= swiftc
SWFLAGS	= -target x86_64-apple-macosx10.11

all: hello vector

%: %.swift
	$(SWIFT) $(SWFLAGS) $< -o $@

clean:
	-rm -f hello
	-rm -f vector
