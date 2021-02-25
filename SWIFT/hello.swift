print("Hello World")
print(CommandLine.arguments)
print(CommandLine.arguments.count)

if CommandLine.arguments.count > 2 {
	print(CommandLine.arguments[1])
	print(CommandLine.arguments[2])
}
