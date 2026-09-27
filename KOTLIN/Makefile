all: p2p.jar nstream.jar stencil.jar transpose.jar

%.jar: %.kt
	kotlinc $< -include-runtime -d $@

clean:
	-rm -f *.jar
