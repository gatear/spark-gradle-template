package spark.examples.supervised

import de.vandermeer.asciitable.AsciiTable
import de.vandermeer.skb.interfaces.document.TableRowStyle
import org.apache.spark.sql.Row

object App {

  implicit private val param: String = "people-example.csv"

  def  main( args: Array[String] ): Unit ={

    //Generating data
    val data = for{ age <- 18d to 50d by 1 } yield 0.0 -> age

    //Making the predictions
    val rows: Array[Row] = new Regression().predict(data)
    val resultTable = new AsciiTable()

    resultTable.addRule()
    resultTable.addRow("Age [Features]","Predicted Salary [LinearRegression]")
    resultTable.addRule()

    rows.foreach{ row: Row => resultTable.addRow( row.getAs[String]("features"),row.getAs[String]("prediction") ) }
    resultTable.addRule()

    println( resultTable.render() )
  }

}
