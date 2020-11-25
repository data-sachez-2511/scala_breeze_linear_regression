package org.apache.spark.ml.male_hw8

import com.google.common.io.Files
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class LinearRegressionTest extends AnyFlatSpec with should.Matchers {
  val delta = 0.0001
//  добавляю программно переменную среды в винду
  System.setProperty("hadoop.home.dir", "C:\\Program Files (x86)\\Hadoop")
  val spark = SparkSession.builder()
    .appName("Simaple App")
    .master("local[4]")
    .getOrCreate()

  val sqlc = spark.sqlContext

  import sqlc.implicits._

  val data: DataFrame = Seq(
    Tuple1(Vectors.dense(2.0, 1.5)),
    Tuple1(Vectors.dense(3.0, 2.5))
  ).toDF("features")

  val trainData: DataFrame = Seq(
    Tuple1(Vectors.dense(0.5, 0.34, 0.5 * 0.1 + 0.34 * 10 - 44.5)),
    Tuple1(Vectors.dense(-0.3, 0.1, -0.3 * 0.1 + 0.1 * 10 - 44.5)),
    Tuple1(Vectors.dense(0.2, -0.7, 0.2 * 0.1 + -0.7 * 10 - 44.5))
  ).toDF("features")

  val validData: DataFrame = Seq(
    Tuple1(Vectors.dense(0.5, 0.34)),
    Tuple1(Vectors.dense(-0.3, 0.1)),
    Tuple1(Vectors.dense(0.2, -0.7))
  ).toDF("features")

  "Model" should "transform input data" in {
    val model = new LinearRegressionModel(
      parameters = Vectors.dense(3.0, 2.0, 1.0).toDense
    ).setInputCol("features").setOutputCol("features")

    val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double](0))

    vectors.length should be(2)

    vectors(0) should be((3.0 * 2.0 + 2.0 * 1.5 + 1.0) +- delta)
    vectors(1) should be((3.0 * 3.0 + 2.0 * 2.5 + 1.0) +- delta)
  }

  "Model" should "transform input data after re-write" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegressionModel(parameters = Vectors.dense(3.0, 2.0, 1.0).toDense)
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).getStages(0).asInstanceOf[LinearRegressionModel]

    val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double](0))

    vectors.length should be(2)

    vectors(0) should be((3.0 * 2.0 + 2.0 * 1.5 + 1.0) +- delta)
    vectors(1) should be((3.0 * 3.0 + 2.0 * 2.5 + 1.0) +- delta)
  }

  "Estimator" should "train on input data" in {
    val estimator = new LinearRegression(lr = 0.001, iterations = 100000).setOutputCol("features").setInputCol("features")

    val model = estimator.fit(trainData)

    val vectors: Array[Double] = model.transform(validData).collect().map(_.getAs[Double](0))

    vectors(0) should be((0.5 * 0.1 + 0.34 * 10 - 44.5) +- delta)
    vectors(1) should be((-0.3 * 0.1 + 0.1 * 10 - 44.5) +- delta)
    vectors(2) should be((0.2 * 0.1 + -0.7 * 10 - 44.5) +- delta)

  }

  "Pipeline" should "train on input data" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression(lr = 0.001, iterations = 100000).setOutputCol("features").setInputCol("features"),
    ))

//    val estimator = new LinearRegression(lr = 0.001, iterations = 100000).setOutputCol("features").setInputCol("features")

    val model = pipeline.fit(trainData).stages(0).asInstanceOf[LinearRegressionModel]

    val vectors: Array[Double] = model.transform(validData).collect().map(_.getAs[Double](0))

    vectors(0) should be((0.5 * 0.1 + 0.34 * 10 - 44.5) +- delta)
    vectors(1) should be((-0.3 * 0.1 + 0.1 * 10 - 44.5) +- delta)
    vectors(2) should be((0.2 * 0.1 + -0.7 * 10 - 44.5) +- delta)

  }

}
