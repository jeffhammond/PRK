all: HelloKt NstreamKt

.PHONY: clean help

%.jar: %.kt
	kotlinc $< -include-runtime -d $@

HelloKt: hello.jar
	kotlin -classpath $< $@

NstreamKt: nstream.jar
	kotlin -classpath $< $@

help:
	echo Run "hello.kt" with "java -jar hello.jar" once hello.jar exists.

clean:
	-rm -f *.exe *.class *.jar

