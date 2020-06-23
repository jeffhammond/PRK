NIM = nim
NIMFLAGS = --showAllMismatches:on

all: nstream nstream_dynamic

nstream: nstream.nim
	$(NIM) compile $(NIMFLAGS) --verbosity:0 --define:iterations=10 --define:length=1000 nstream.nim

nstream_dynamic: nstream_dynamic.nim
	$(NIM) compile $(NIMFLAGS) --verbosity:0 nstream_dynamic.nim

clean:
	-rm -f hello p2p nstream dgemm stencil transpose sparse
	-rm -f nstream_dynamic
