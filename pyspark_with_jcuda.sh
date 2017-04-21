#!/bin/bash

# Starts a PySpark shell with JCuda dependecies
os_suffix='linux-x86_64'
version='0.8.0'
jars='.'
if [[ -z "${SPARK_HOME}" ]]; then
	echo "Error: Expected SPARK_HOME to be set"
else
	for lib in jcuda jcublas jcufft jcusparse jcusolver jcurand jnvgraph jcudnn
	do
		file=$lib'-'$version'.jar'
		if [ ! -f $file ]; then
			url='https://search.maven.org/remotecontent?filepath=org/jcuda/'$lib'/'$version'/'$file
			wget -O $file $url 
		fi
		jars=$jars':'$file

		file=$lib'-natives-'$version'-'$os_suffix'.jar'
		if [ ! -f $file ]; then
			url='https://search.maven.org/remotecontent?filepath=org/jcuda/'$lib'-natives/'$version'/'$file
			wget -O $file $url
		fi
		jars=$jars':'$file
	done
	$SPARK_HOME/bin/pyspark --driver-class-path $jars
fi
