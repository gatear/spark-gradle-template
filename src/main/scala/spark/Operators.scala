package spark

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.{Row, RowFactory}

object Operators {

  //Overloading operators for easy row,vector manipulation
  def ~~>[T](that: Vector[T]): Row = {
    RowFactory.create(that)
  }

  def \\>(that: Traversable[Double]): DenseVector = {
    new DenseVector( that.toArray )
  }

  def ||>(that: Array[Double]): SparseVector = {
    val nonZeroThat = that.zipWithIndex.filter(_._1 != 0.0)
    new SparseVector(that.length, nonZeroThat.map(t => t._2), nonZeroThat.map(t => t._1))
  }

}
