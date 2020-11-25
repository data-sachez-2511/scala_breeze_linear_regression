package org.apache.spark.ml.male_hw8

import breeze.linalg.{DenseMatrix, sum, DenseVector => DenseVectorBreeze}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReader,, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(str: String): this.type = set(inputCol, str)

  def setOutputCol(str: String): this.type = set(outputCol, str)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    if (schema.fieldNames.contains($(inputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression private[male_hw8](override val uid: String,
                                         val lr: Double,
                                         val iterations: Int) extends Estimator[LinearRegressionModel]
                                                              with LinearRegressionParams {

  private[male_hw8] def this(lr: Double, iterations: Int) = this(Identifiable.randomUID("linearRegression"),
                                                                                                    lr, iterations);

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val temp = dataset.select(dataset($(inputCol))).rdd.map(_.getAs[Vector]("features")).map(x => x.asBreeze).collect()
    val N = temp.length
    val M = temp(0).length - 1
    val matrix = DenseMatrix.rand[Double](N, M)
    val Y = DenseVectorBreeze.rand[Double](N)
    for (i <- 0 until N) {
      for (j <- 0 until M) {
        matrix(i, j) = temp(i)(j)
      }
      Y(i) = temp(i)(M)
    }
    val weights = DenseVectorBreeze.rand[Double](M + 1)
    for (i <- 0 to iterations) {
      val currentY = (matrix * weights(0 until M)) + DenseVectorBreeze.fill(N) {
        weights(-1)
      }
      val bGradient = sum(currentY - Y) / N
      for (j <- 0 until M) {
        val mGradient = sum(matrix(::, j) * (currentY - Y)) / N
        weights(j) = weights(j) - lr * mGradient
      }
      weights(M) = weights(M) - lr * bGradient
    }
    copyValues(new LinearRegressionModel(
      parameters = Vectors.fromBreeze(weights).toDense
    ).setInputCol("features").setOutputCol("features")).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = copyValues(new LinearRegression(lr,
                                                                                                         iterations))

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

class LinearRegressionModel private[male_hw8](override val uid: String,
                                              val parameters: DenseVector
                                             ) extends Model[LinearRegressionModel] with LinearRegressionParams
                                                                                    with MLWritable {


  private[male_hw8] def this(uid: String) = this(uid, Vectors.zeros(0).toDense);


  private[male_hw8] def this(parameters: DenseVector) = this(Identifiable.randomUID("linearRegressionModel"), parameters);

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(parameters))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform", (x: Vector) => {
      sum(x.asBreeze * parameters.asBreeze(0 until parameters.size - 1)) + parameters.asBreeze(-1)
    })
    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))));
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(Seq(Tuple1(parameters.asInstanceOf[Vector]))).write.parquet(path + "/params")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val data = sqlContext.read.parquet(path + "/params")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val params_ = data.select(data("_1").as[Vector]).first().toDense

      val model = new LinearRegressionModel(params_)
      metadata.getAndSetParams(model)
      model
    }
  }
}
