#!/bin/bash

echo "Adding required JARs to classpath and compiling sources..."
javac -classpath ./:./lib/commons-logging-1.1.2.jar:./lib/weka.jar:./lib/hadoop-1.2.1/hadoop-core-1.2.1.jar ./src/Run.java

if [ $? -eq 0 ]
then
	echo "Compilation Successful"
fi

echo "Copying class files..."
mkdir -p src/output
cp src/*.class src/output/

echo "Building JAR file..."
pushd src/
jar -cvf weka-test.jar -C output/ .

if [ $? -eq 0 ]
then
	echo "JAR created successfully"
fi
popd

echo "Complete."

# Run the JAR here? Just don't lost this command...
hadoop jar weka-test.jar Run 10 weka.classifiers.trees.J48 ./slug.arff output/
