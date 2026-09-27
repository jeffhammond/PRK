# How to build

Just type `make`

Each kernel is compiled with `kotlinc -include-runtime`, which bundles the
Kotlin stdlib into the jar, so the result is directly runnable with `java -jar`
(no need for a separate `kotlin` runtime install or `-classpath` setup).

# How to run

```
java -jar p2p.jar 10 1024 1024
java -jar nstream.jar 10 10000000
java -jar stencil.jar 10 1000
java -jar stencil.jar 10 1000 star 4
java -jar stencil.jar 10 1000 grid 2
java -jar transpose.jar 10 1000
java -jar transpose.jar 10 1000 32
```
