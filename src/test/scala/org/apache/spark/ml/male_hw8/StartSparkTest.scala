package org.apache.spark.ml.male_hw8
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.sql.SparkSession

class StartSparkTest extends AnyFlatSpec with should.Matchers {
  "Spark" should "start context" in {
    System.setProperty("hadoop.home.dir", "C:\\Program Files (x86)\\Hadoop")
    val spark = SparkSession.builder()
      .appName("Simaple App")
      .master("local[4]")
      .getOrCreate()

    Thread.sleep(60000)
  }

}
