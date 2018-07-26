package spark.examples

import org.apache.spark.ml.linalg.{DenseVector, Matrix, SparseVector}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{Row, RowFactory}
import spark.SparkEnv


object Example2 extends SparkEnv {

  //Overloading operators for easy row,vector manipulation
  //https://spark.apache.org/docs/2.2.0/ml-statistics.html

  def ~~>[T](that: Vector[T]): Row = {
    RowFactory.create(that)
  }

  def \\>(that: Array[Double]): DenseVector = {
    new DenseVector(that)
  }

  def ||>(that: Array[Double]): SparseVector = {
    val nonZeroThat = that.zipWithIndex.filter(_._1 != 0.0)
    new SparseVector(that.length, nonZeroThat.map(t => t._2), nonZeroThat.map(t => t._1))
  }


  def run() = {

    import spark.implicits._
    val version = spark.version
    println("SPARK VERSION = " + version)

    val data = Seq(
      ||>(Array(0.0, 1.0, 1.0, 4.0, 4.0)),
      \\>(Array(1.0, 9.0, 7.0, 0.0, 0.0)),
      ||>(Array(0.0, 0.0, 8.0, 8.0, 8.0))
    )

    val df = data.map(Tuple1.apply).toDF("features")

    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head

    //String interpolation
    println(s"Pearson correlation matrix:\n $coeff1")

    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head

    //String interpolation
    println(s"Spearman correlation matrix:\n $coeff2")

  }
}
