#!/bin/bash
# Author: Trevor Hodde
#
# This script is responsible for the following:
# 	- Upload the dataset to HDFS
#	- Remove old data
# 	- Update the classpath (if necessary)
# 	- Compiles the sources
# 	- Builds the JAR file to run the program
# 	- Executes the program
#

declare DATASET=${1:-player_regular_season.txt}

# Remove old data set
echo "Removing old data set..."
hadoop fs -rmr /home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/$DATASET

# Upload most recent data set
echo "Uploading latest data set..."
hadoop fs -put bin/$DATASET /home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/

# Remove old output if it exists
echo "Removing old output directory..."
hadoop fs -rmr /home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/output

echo "Adding required JARs to classpath and compiling sources..."
export CLASSPATH=./:./lib/commons-logging-1.1.2.jar:./lib/weka.jar:./lib/hadoop-1.2.1/hadoop-core-1.2.1.jar
javac -classpath $CLASSPATH ./src/Run.java

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

# return to our previous directory
popd

echo "Complete."

pushd src/

echo "Executing program."
chmod +x weka-test.jar
# Run the JAR here? Just don't lost this command...
hadoop jar weka-test.jar Run 10 weka.classifiers.bayes.NaiveBayes /home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/$DATASET /home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/output/ 
