import pyspark
from pyspark.sql import SparkSession
import sys
import tempfile
import time
import pandas

spark = SparkSession.builder \
                    .master("local[*]") \
                    .config("spark.executor.memory", "8g") \
                    .config("spark.driver.memory", "8g") \
                    .appName("findShortestPaths") \
                    .getOrCreate()

spark.sparkContext.setCheckpointDir("checkpoints")

def find_paths_doubling(inFile, outFile):

    edgesDF = spark.read.format("csv") \
    .option("sep", ",") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load(inFile)

    newColumns = ["path_1","path_2","length"]
    pathsDF = edgesDF.toDF(*newColumns)

    pathsDF.createOrReplaceTempView("paths")
    pathsDF.createOrReplaceTempView("doubled")

    oldDF = pathsDF
    newDF = pathsDF

    doubledDF_len = len(newDF.head(1))

    i = 0
    first_iter = True
    while first_iter or doubledDF_len > 0:
        time_it_s = time.time()
        if not first_iter:
            oldDF = newDF

            doubledDF = spark.sql("select P1.path_1 as path_1, P2.path_2 as path_2, min(P1.length + P2.length) as length \
                                from doubled as P1 join doubled as P2 on P1.path_2 = P2.path_1 group by P1.path_1, P2.path_2").checkpoint()

            extendedDF = spark.sql("select P1.path_1 as path_1, P2.path_2 as path_2, min(P1.length + P2.length) as length \
                                from doubled as P1 join paths as P2 on P1.path_2 = P2.path_1 group by P1.path_1, P2.path_2").checkpoint()

            extendedDF = oldDF.union(extendedDF).cache()
            extendedDF.createOrReplaceTempView("extended")
            extendedDF = spark.sql("select path_1, path_2, min(length) as length from extended group by path_1, path_2").checkpoint()

            newDF = extendedDF.union(doubledDF).cache()
            newDF.createOrReplaceTempView("paths")
            newDF = spark.sql("select path_1, path_2, min(length) as length from paths group by path_1, path_2").checkpoint()

            doubledDF = newDF.subtract(extendedDF).checkpoint()
            newPathsDF = newDF.subtract(doubledDF).checkpoint()
            newPathsDF.createOrReplaceTempView("paths")
            doubledDF.createOrReplaceTempView("doubled")
            doubledDF_len = len(doubledDF.head(1))

        else:
            first_iter = False

            doubledDF = spark.sql("select P1.path_1 as path_1, P2.path_2 as path_2, min(P1.length + P2.length) as length \
                                from doubled as P1 join doubled as P2 on P1.path_2 = P2.path_1 group by P1.path_1, P2.path_2").checkpoint()

            newDF = oldDF.union(doubledDF).cache()
            newDF.createOrReplaceTempView("paths")
            newDF = spark.sql("select path_1, path_2, min(length) as length from paths group by path_1, path_2").checkpoint()

            doubledDF = newDF.subtract(oldDF).checkpoint()
            newPathsDF = newDF.subtract(doubledDF).checkpoint()
            newPathsDF.createOrReplaceTempView("paths")
            doubledDF.createOrReplaceTempView("doubled")
            doubledDF_len = len(doubledDF.head(1))

        print(f"TIME ELAPSED IN ITERATION {i}:", time.time() - time_it_s)
        i+=1

    pandasDF = newDF.coalesce(1).toPandas()
    pandasDF.to_csv(outFile, index=False)


def find_paths_linear(inFile, outFile):

    edgesDF = spark.read.format("csv") \
    .option("sep", ",") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load(inFile)

    newColumns = ["path_1","path_2","length"]
    extDF = edgesDF.toDF(*newColumns)

    edgesDF.createOrReplaceTempView("edges")
    edgesDF = edgesDF.cache()

    oldDF = extDF.cache()
    extDF.createOrReplaceTempView("exts")
    extDF_len = len(extDF.head(1))

    i = 0
    first_iter = True
    while first_iter or extDF_len > 0:
        time_it_s = time.time()
        if not first_iter:
            oldDF = newDF.checkpoint()
        else:
            first_iter = False

        extDF = spark.sql("select exts.path_1 as path_1, edges.edge_2 as path_2, min(exts.length + edges.length) as length \
                        from exts join edges on exts.path_2 = edges.edge_1 group by exts.path_1, edges.edge_2").checkpoint()

        newDF = oldDF.union(extDF)
        newDF.createOrReplaceTempView("paths")
        newDF = spark.sql("select path_1, path_2, min(length) as length from paths group by path_1, path_2").cache()
        extDF = newDF.subtract(oldDF)
        extDF.createOrReplaceTempView("exts")
        
        extDF_len = len(extDF.head(1))

        print(f"TIME ELAPSED IN ITERATION {i}:", time.time() - time_it_s)
        i+=1


    pandasDF = newDF.coalesce(1).toPandas()
    pandasDF.to_csv(outFile, index=False)


def find_paths(algorithm, inFile, outFile):
    if algorithm == "linear":
        find_paths_linear(inFile, outFile)
    elif algorithm == "doubling":
        find_paths_doubling(inFile, outFile)


t_s = time.time()
algorithm, inFile, outFile = sys.argv[1], sys.argv[2], sys.argv[3]
print(algorithm, inFile, outFile)
# algorithm, inFile, outFile = "linear", "./in_con.csv", "./out_con.csv"
find_paths(algorithm, inFile, outFile)
t_e = time.time()
print(t_e - t_s)