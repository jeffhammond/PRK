AC = gnat
ACFLAGS = make

all: hello args nstream p2p stencil transpose

%: %.adb
	$(AC) $(ACFLAGS) $< -o $@

clean:
	-rm -f *.i *.o *.ali
	-rm -f hello args nstream p2p stencil transpose

