SWIFT 	= swiftc
#SWFLAGS	= -target x86_64-apple-macosx10.11

all: hello vector nstream

%: %.swift
	$(SWIFT) $(SWFLAGS) $< -o $@

clean:
	-rm -f hello
	-rm -f vector
	-rm -f nstream
