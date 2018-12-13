/*
Data can be downloaded from the following website:
http://projects.knmi.nl/klimatologie/daggegevens/selectie.cgi

Calculating different model evaluation metrics
Parameter tuning using k-fold and  gird search.
Definition of a base-line model
For random forest use more than two states for rainfall condition by changing bucketizer parms.


ParamMap is a container for ParamPairs
Param are refered thtiugh an instantiated object!
 */
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, VectorAssembler}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window

import scala.util.Random
object DTree {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DecisionTree")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

   import spark.implicits._



    // Create input dataframe from data file
    val dtInstance = new DTree(spark)
    val laggedDataFrame = dtInstance.createDataFrame()
    laggedDataFrame.show(10,0)

    // Splits input dataframe to training and test sets
    val Array(trainData, testData) = laggedDataFrame.randomSplit(Array(0.7,0.3))
    trainData.cache()
    testData.cache()

    // Transforming continious label (rainfall data) to bucketed labels
    val bucketizer = new Bucketizer()
      .setInputCol("RH")
      .setOutputCol("WetDry")
      .setSplits(Array(Double.NegativeInfinity, 0.1, Double.PositiveInfinity))
    //.transform()

    val countWetDry = bucketizer.transform(laggedDataFrame).groupBy("WetDry").count()
    countWetDry.show()


    // Preparing input vector
    val inputCols = laggedDataFrame.columns.filter{colName=>
      colName.startsWith("Lag") &&
        !colName.contains("Date")}

    //for (elm<-inputCols) println(elm)

    // Assembling input variables
    val inputAssembler = new VectorAssembler()
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
      .setStages(Array(bucketizer, inputAssembler, decisionTree))

    /*

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

   */

    // Tuning hyper parameters

    // Defining a search grid
    val parmGrid = new ParamGridBuilder()
      .addGrid(decisionTree.impurity, Seq("gini", "entropy"))
      .addGrid(decisionTree.maxDepth, Seq(2,3)) //,5,7,10)
      .addGrid(decisionTree.minInfoGain, Seq(0.01,0.05)) //0.06, 0.08, 0.1
      .build()

    // Defining a model evaluator
    val metricEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("WetDry")
      .setPredictionCol("Prediction")
      .setMetricName("accuracy")


    val modelTuner = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(metricEvaluator)
      .setEstimatorParamMaps(parmGrid)
      .setCollectSubModels(false)
      .setTrainRatio(0.9)


    val tunedModel = modelTuner.fit(trainData)


    // Printing combination of different hyper parameters and their evaluation metrics for validation data
    val validMetricAndHyperParam = tunedModel.getEstimatorParamMaps
      .zip(tunedModel.validationMetrics).sortBy(-_._2)
      .foreach{case (paramMap:ParamMap, valMetric:Double) =>
        println(s"Model accuracy is ${valMetric} for the following hyper paramers: \n ${paramMap}")}


    // Getting the best model (DecisionTreeClassificationModel) and its parameters
    val bestDtreeModel:DecisionTreeClassificationModel = tunedModel
      .bestModel
      .asInstanceOf[PipelineModel]
      .stages.last
      .asInstanceOf[DecisionTreeClassificationModel]


    // Most influential input features for the train deception tree model
    val topFeatures:linalg.Vector = bestDtreeModel
      .featureImportances

    val df:DataFrame = inputCols
      .zip(topFeatures.toArray)
      .toList.toDF("Features", "Importance")
      .sort($"Importance".desc)

    df.show()

    // Printing best model's (hyper-) parameters
    val parmMap = bestDtreeModel.extractParamMap()
      .toSeq.map{paramPair=>
      (paramPair.param.toString().split("__").apply(1), paramPair.value)}
      .foreach(println(_))


    // Getting the best's fitted parameters



    // Best model's evaluation metric





    val res:linalg.Vector = bestDtreeModel.featureImportances

    val struc = bestDtreeModel.toDebugString

    val numfeeature = bestDtreeModel.numFeatures


    println(numfeeature)


    println(res)

    println(struc)






















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
    // Loading data from a .csv file
    val weatherData = spark.read
      .option("comment","#")
      .csv(path)
      .toDF(colNames:_*)
      .select(dailyCols.head, dailyCols.tail:_*)
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

    // Lagging weather data and removing redundant columns
    val window = Window.orderBy("Date")
      .partitionBy("STN")

    val laggedDataFrame = weatherData
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
      .na.drop() // feature vales cannot be null!
      // You may drop redundant columns for saving memory
      .select("Date", "STN", "LagDate", "LagDDVEC", "LagFHVEC", "LagFG",
      "LagTG", "LagSQ", "LagSP", "LagQ", "LagPG", "LagNG", "LagUG", "LagEV24",
      "LagRH", "RH")
    laggedDataFrame

  }


  def dTree(): Unit = {
    /* save model, show metrics
    This model makes a binary prediction using the wet threshold value.
    splitRatios: Array[Double]
    wetThreshold: Double
    */
    ???}

  def dTreeHyperTuned() : Unit = {???}

  def rndForest() : Unit = {
    /*
    This model uses rainfall distribution for detemining three or four level of rainfall intensity.
     */

    ???}

  def rndForestHyperTuned() : Unit = {???}




}




