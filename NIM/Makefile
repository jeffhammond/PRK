NIM = nim

all: hello

hello: hello.nim
	$(NIM) compile --verbosity:0 hello.nim

clean:
	-rm -f hello p2p nstream dgemm stencil transpose sparse
