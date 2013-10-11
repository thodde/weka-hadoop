weka_hadoop
===========

Using the Weka Machine Learning libraries on the Hadoop platform
The goal of this system is to try to use the Hadoop platform to perform cross validation of a machine learning algorithm. This system will take an input file (in ARFF format), distribute the file to the selected number of nodes and run a machine learning algorithm on each node. Using this system it is possible to run all cross validation processes at once instead of looping through the data. The system has been tested with Naive Bayes and IBk.

