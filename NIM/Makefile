NIM = nim
NIMFLAGS = --showAllMismatches:on

all: nstream

nstream: nstream.nim
	$(NIM) compile $(NIMFLAGS) --verbosity:0 --define:iterations=10 --define:length=1000 nstream.nim

clean:
	-rm -f hello p2p nstream dgemm stencil transpose sparse
