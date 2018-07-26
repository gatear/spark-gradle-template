package spark.examples

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import spark.SparkEnv

object Example1 extends SparkEnv {

  final case class Person(firstName: String, lastName: String, country: String, age: Int, salary: Int)

  def run () = {

    import spark.implicits._
    // Prepare training documents from a list of (id, text, label) tuples.
    val training = reader.csv("people-example.csv").as[Person].map( p => p.age -> Vectors.dense( p.salary)).toDF("label", "features")
    println(s"The training dataframe columns are: ${training.columns}")

    // Create a LogisticRegression instance. This instance is an Estimator.
    val lr = new LogisticRegression()

    // Print out the parameters, documentation, and any default values.
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    // We may set parameters using setter methods.
    lr.setMaxIter(10)
      .setRegParam(0.01)

    // Learn a LogisticRegression model. This uses the parameters stored in lr.

    val model = lr.fit(training)
    // Since model is a Model (i.e., a Transformer produced by an Estimator),
    // we can view the parameters it used during fit().
    // This prints the parameter (name: value) pairs, where names are unique IDs for this
    // LogisticRegression instance.

    println("Model was fitted using parameters: " + model.parent.extractParamMap)

    // Prepare test data.
    val test = spark.createDataFrame(Seq(
      (25.0, Vectors.dense(500.0)),
      (30.0, Vectors.dense(3000.0)),
      (40.0, Vectors.dense(4000.0))
    )).toDF("label", "features")

    // Make predictions on test data using the Transformer.transform() method.
    // LogisticRegression.transform will only use the 'features' column.

    model.transform(test)
         .select("features", "label", "probability", "prediction")
         .collect()
         .foreach { case Row(features, label, prob, prediction) =>
        println(s"($features, $label) prediction=$prediction")
      }

  }
}
