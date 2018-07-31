package spark.examples.supervised

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import spark.SparkEnv
import spark.Operators._

//self typing a trait to avoid misusing it
sealed trait Linear {
  this :Regression =>
  def predict( ts :  IndexedSeq[(Double, Double)] ): Array[Row]
}

case class Person(firstName: String, lastName: String, country: String, age: Int, salary: Double)

class Regression()(implicit csvFile: String) extends Linear with SparkEnv{

  import spark.implicits._
  var points: DataFrame = reader.csv(csvFile)
                                .as[Person]
                                .map( p => (p.salary, \\>(p.age::Nil) ))
                                .toDF("label", "features")

  val lr = new LinearRegression()
                      .setMaxIter( 100)
                      .setRegParam(0.3)
                      .setElasticNetParam(0.8)

  val model = lr.fit(points)
  println(s"Model Summary: ${ model.summary.totalIterations } total iterations")

  override def predict(ts: IndexedSeq[(Double, Double)] ): Array[Row] ={

    val data: Dataset[_] = spark.createDataFrame (ts.map( t => ( t._1, \\>(t._2::Nil)  ))).toDF("label","features")

    model.transform( data )
         .select("features", "prediction")
         .collect()
  }

}

