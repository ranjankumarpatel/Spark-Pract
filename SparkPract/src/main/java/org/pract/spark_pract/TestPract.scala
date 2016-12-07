package org.pract.spark_pract

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object TestPract extends App {
  
  val dir = "file:///D:/git/Spark-Pract/scala-pract/spark-warehouse"
  val conf = new SparkConf().setAppName("chkspark").setMaster("local[*]")
  val sc = new SparkContext(conf)
  
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  println(sc)
  
  println(sqlContext)
  val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .load(dir+"/iris.csv")
  sc.textFile(dir+"/iris.csv", 1).map { x => x.split(",") }
  val df1 = sqlContext.read.csv(dir+"/iris.csv")
  println(df)
}