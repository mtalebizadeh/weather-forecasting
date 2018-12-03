/*
Including multiple weather station (Sort by Date grouped by station)
Using different time lags
Calculating different model evaluation metrics
Parameter tuning using k-fold and  gird search.
Definition of a base-line model
For random forest use more than two states for rainfall condition by changing bucketizer parms.
 */
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import scala.util.Random
object DTree {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DecisionTree")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val dtInstance = new DTree(spark)
    val weatherDataFrame = dtInstance.createDataFrame()
    //weatherDataFrame.show(5,0)

    // Lagging the weatherdataFrame...
    val window = Window.orderBy("Date")
      .partitionBy("STN")
    val laggedDataFrame = weatherDataFrame
      .withColumn("LagDate", functions.lag($"Date", 1).over(window))
      .withColumn("LagDDVEC", functions.lag("DDVEC", 1).over(window))
      .withColumn("LagFHVEC", functions.lag("FHVEC", 1).over(window))
      .withColumn("LagFG", functions.lag("FG", 1).over(window))
      .withColumn("LagTG", functions.lag("TG", 1).over(window))
      .withColumn("LagSQ", functions.lag("SQ", 1).over(window))
      .withColumn("LagSP", functions.lag("SP", 1).over(window))
      .withColumn("LagQ", functions.lag("Q", 1).over(window))
      .withColumn("LagPG", functions.lag("PG", 1).over(window))
      .withColumn("LagNG", functions.lag("NG", 1).over(window))
      .withColumn("LagUG", functions.lag("UG", 1).over(window))
      .withColumn("LagEV24", functions.lag("EV24", 1).over(window))
      .withColumn("LagRH", functions.lag("RH", 1).over(window))
      .na.drop()  //Values to assemble cannot be null!
      // You may drop redundant columns for saving memory
      .select("LagDate", "LagDDVEC", "LagFHVEC", "LagFG",
       "LagTG", "LagSQ", "LagSP", "LagQ", "LagPG", "LagNG", "LagUG", "LagEV24",
       "LagRH", "RH", "Date", "STN")

    laggedDataFrame.show(10,0)
    val Array(trainData, testData) = laggedDataFrame.randomSplit(Array(0.7,0.3))
    trainData.cache()
    testData.cache()

    // Transforming continious label (rainfall data) to bucketed labels
    val bucketizer = new Bucketizer()
      .setInputCol("RH")
      .setOutputCol("WetDry")
      .setSplits(Array(Double.NegativeInfinity, 0.1, Double.PositiveInfinity))
    //.transform()

    val countWet = bucketizer.transform(laggedDataFrame).groupBy("WetDry").count()

    countWet.show()


    // Preparing input vector
    val inputCols = laggedDataFrame.columns.filter{colName=>
      colName.startsWith("Lag") &&
        !colName.contains("Date")}

    //for (elm<-inputCols) println(elm)

    // Assembling input variables
    val inputVector = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("inputVector")
    //.transform()


    // Creating a decesion tree model using inputVector and label variables (i.e., WetDry)
    val decisionTree = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("inputVector")
      .setLabelCol("WetDry")
      .setPredictionCol("Prediction")
    //.fit()

    // Creating pipeline model and setting stages:
    val pipeline = new Pipeline()
      .setStages(Array(bucketizer, inputVector, decisionTree))

    // Fitting the pipeline model on trainData
    val pipelineModel = pipeline.fit(trainData)

    // Model predictions for training and test data
    val trainDataPredictions = pipelineModel.transform(trainData)
    val testDataPredictions = pipelineModel.transform(testData)


    // Calculating model evaluation metrics on training and test data
    val metricEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("WetDry")
      .setPredictionCol("Prediction")
      .setMetricName("f1")

    val trainEvalMetric = metricEvaluator.evaluate(trainDataPredictions)
    val testEvalMetric = metricEvaluator.evaluate(testDataPredictions)

    println(s"Evaluation metric for training data is ${trainEvalMetric} " +
      s"and for test data is ${testEvalMetric}")



    trainData.unpersist()
    testData.unpersist()

    println("End of the cat and mouse!")
    spark.close()

  }
}
class DTree(private val spark: SparkSession) {

  import spark.implicits._

  def createDataFrame(path: String="/home/mansour/IdeaProjects/predictive_ML/Data/KNMI_20181202_All_Data.txt"): Dataset[Row] = {

    val colNames = Seq(
      "STN", "Date","DDVEC", "FHVEC", "FG",
      "FHX", "FHXH", "FHN", "FHNH", "FXX",
      "FXXH", "TG", "TN", "TNH", "TX",
      "TXH", "T10N", "T10NH", "SQ", "SP",
      "Q", "DR", "RH", "RHX", "RHXH",
      "PG", "PX", "PXH", "PN", "PNH",
      "VVN", "VVNH", "VVX", "VVXH", "NG",
      "UG", "UX", "UXH", "UN", "UNH", "EV24"
    )

    val dailyCols = Seq(
      "STN", "Date", "DDVEC", "FHVEC", "FG", "TG", "SQ",
      "SP", "Q", "PG", "NG", "UG", "EV24", "RH"
    )

    val weatherData = spark.read
      .option("comment","#")
      .csv(path)
      .toDF(colNames:_*)
      .select(dailyCols.head, dailyCols.tail:_*)
      //.withColumn("Date", functions.to_date($"Date","yyyymmdd") )
      .withColumn("DDVEC", $"DDVEC".cast("double"))
      .withColumn("FHVEC", $"FHVEC".cast("double"))
      .withColumn("FG", $"FG".cast("double"))
      .withColumn("TG", $"TG".cast("double"))
      .withColumn("SQ", $"SQ".cast("double"))
      .withColumn("SP", $"SP".cast("double"))
      .withColumn("Q", $"Q".cast("double"))
      .withColumn("PG", $"PG".cast("double"))
      .withColumn("NG", $"NG".cast("double"))
      .withColumn("UG", $"UG".cast("double"))
      .withColumn("EV24", $"EV24".cast("double"))
      .withColumn("RH", $"RH".cast("double"))
    //.na.drop()
    weatherData

  }

}




