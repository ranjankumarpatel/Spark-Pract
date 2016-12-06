package org.scala.pract.algo
import scala._

object Pract1 extends App {
  //println(io.Source.stdin.getLines().take(2).map(_.toInt).sum)
  
   val size = scala.io.StdIn.readInt
   println(io.Source.stdin.getLines().take(2).map(_.toInt).sum)
}