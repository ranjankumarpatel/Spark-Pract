package org.scala.pract.spark

import org.apache.spark._
import org.apache.spark.sql.SQLContext


object IrisDecision extends App {

  val dir = "D:/git/Spark-Pract/scala-pract/spark-warehouse"

  val conf = new SparkConf().setAppName("chkspark").setMaster("local[*]")
  val sc = new SparkContext(conf)

  val sqlContext = new SQLContext(sc);
  import sqlContext.implicits._

  case class Iris(Sepal_Length: Float, Sepal_Width: Float, Petal_Length: Float, Petal_Width: Float, Species: String)
  val dataLines = sqlContext.read.csv(dir + "/iris.csv")

  print(dataLines)

}