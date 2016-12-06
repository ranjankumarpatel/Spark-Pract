package org.scala.pract

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object sparkSc {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("chkspark").setMaster("local[*]")
    val sc = new SparkContext(conf)
    println(sc)

    Thread.sleep(1000);

    val data = Array(1, 2, 3, 4, 5)
    val distData = sc.parallelize(data)
    
    println(distData)
  }
}