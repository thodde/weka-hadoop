#!/bin/bash

echo "Adding required JARs to classpath and compiling sources..."
javac -classpath ./:./lib/commons-logging-1.1.2.jar:./lib/weka.jar:./lib/hadoop-1.2.1/hadoop-core-1.2.1.jar ./src/Run.java

if [ $? -eq 0 ]
then
	echo "Compilation Successful."
fi
