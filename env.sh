module load spark
module load python/3.5.1

# my added lines:
setenv SPARK_HOME /encs/pkg/spark-2.2.0/root
setenv PYTHONPATH ${SPARK_HOME}/python/:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip
setenv PYTHONPATH ${PYTHONPATH}:./answers
setenv PATH ${PATH}:${HOME}/.local/bin



