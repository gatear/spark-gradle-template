package spark.examples

object Main {

  def main( args: Array[String] ): Unit ={

    args.head match {
      case "1" => Example1.run()
      case "2" => Example2.run()
    }

  }

}
