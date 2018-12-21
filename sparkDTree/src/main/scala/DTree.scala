/*

Data can be downloaded from the following website:
http://projects.knmi.nl/klimatologie/daggegevens/selectie.cgi

Calculating different model evaluation metrics
Parameter tuning using k-fold and  gird search.
Definition of a base-line model
For random forest use more than two states for rainfall condition by changing bucketizer parms.


ParamMap is a container for ParamPairs
Param are refered thtiugh an instantiated object!
Key terms: Ensemble simulation, Decision Tree, Random Forest, Hyper parameter tuning, weather forecast.
           Input sensitivty analysis, Descretie Time series analysis, Pipeline model.

 */
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, QuantileDiscretizer, VectorAssembler}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, linalg}
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
    /*
    val countWetDry = bucketizer.transform(laggedDataFrame).groupBy("WetDry").count()
    countWetDry.show()
    */
      dtInstance.dTree(trainData, testData)
    //dtInstance.dTreeHyperTuned(trainData, testData)
    //dtInstance.rndForest(trainData, testData)
    //dtInstance.rndForestHyperTuned(trainData, testData)

    trainData.unpersist()
    testData.unpersist()
    println("End of the analysis!")
    spark.close()
  }
}
class DTree(private val spark: SparkSession) {

  import spark.implicits._
  /** Creates dataframe from text file containing the Netherlands weather stations.
    *
    * @param path Fila path containing weather data
    * @return Dataframe containing inputs and label data
    */
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
    val removeMinusRH = functions.udf((RH:Double)=> if(RH>=0) RH else Double.NaN)
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
      //.withColumn("RH",  $"RH".cast("double"))
      .withColumn("RH", removeMinusRH($"RH".cast("double")))

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

  /** Assembles features and label and building a pipeline
    *
    * @param inputCols
    * @return
    */
  def createFeatureAndLabelPipeline(inputCols: Array[String]):Pipeline = {

    val bucketizer = new Bucketizer()
      .setInputCol("RH")
      .setOutputCol("WetDry")
      .setSplits(Array(Double.NegativeInfinity, 0.1, Double.PositiveInfinity))
    //.transform()

    val quantileDiscretizer = new QuantileDiscretizer()
      .setInputCol("RH")
      .setOutputCol("WetDry")
      .setNumBuckets(3)

    // Assembling input vector
    val inputAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("inputVector")
    //.transform()

    val pipeline = new Pipeline()
      .setStages(Array(quantileDiscretizer, inputAssembler)) // first stage could also be a bucketizer

    pipeline
  }

  /** Calculates fouer model evaluation metrics.
    *
    * @param dataFrame
    * @param predCol
    * @param labelCol
    * @param modelName
    * @param dataFrameName
    */
  def evaluateClassificationModel(dataFrame:DataFrame,
                                  predCol:String = "Prediction",
                                  labelCol:String = "WetDry",
                                  modelName:String = "",
                                  dataFrameName:String = "data") = {

    // Calculating model evaluation metrics on training and test data
    val metricEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predCol)
    //.setMetricName("f1")

    val evalMetricNames = Seq("f1", "accuracy", "weightedPrecision", "weightedRecall")
    val evalMetricValues4TrainData  = evalMetricNames
      .map(metricEvaluator.setMetricName(_).evaluate(dataFrame))

    val evalMetricValues4TestData = evalMetricNames
      .map(metricEvaluator.setMetricName(_).evaluate(dataFrame))

    println(s"Prediction evaluation metrics for ${modelName} model for ${dataFrameName} are:" )
    evalMetricNames.zip(evalMetricValues4TrainData).foreach(println(_))
    println()
  }

  /** Creates a decision tree pipeline.
    *
    * @param inputCols column names representing input features
    * @return a decision treee pipeline
    */
  def createDTreePipeline(inputCols: Array[String]):Pipeline = {

    val featureAndLabelStage = createFeatureAndLabelPipeline(inputCols)

    // Creating a decision tree model using assembled feature Vector and label variables.
    val decisionTree = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setMinInstancesPerNode(50)
      .setFeaturesCol("inputVector")
      .setLabelCol("WetDry")
      .setPredictionCol("Prediction")
    //.fit()
    // Creating pipeline model and setting stages:
    val pipeline = new Pipeline()
      .setStages(Array(featureAndLabelStage, decisionTree))

    pipeline
  }


  /** Train and evaluates a decision tree model for predicting precipitation states.
    *
    * @param trainData train dataset
    * @param testData test dataset
    */
  def dTree(trainData:DataFrame, testData:DataFrame): Unit = {

    // Collecting input column names.
    val inputCols = trainData.columns.filter{colName=>
      colName.startsWith("Lag") &&
        !colName.contains("Date")}

    // Creating a decision tree pipeline
    val pipeline = createDTreePipeline(inputCols)

    // Fitting the pipeline on trainData to create a model
    val pipelineModel = pipeline.fit(trainData)

    // Calculating model predictions for training and test data
    val trainDataPredictions = pipelineModel.transform(trainData)
    val testDataPredictions = pipelineModel.transform(testData)

    evaluateClassificationModel(dataFrame = trainDataPredictions,
      modelName = "Decision Tree",
      dataFrameName = "trainData")

    evaluateClassificationModel(dataFrame = testDataPredictions,
      modelName = "Decision Tree",
      dataFrameName = "testData")

    // Most influential input features for the decision tree model
    val topFeatures:linalg.Vector = pipelineModel.stages.last
      .asInstanceOf[DecisionTreeClassificationModel].featureImportances

    val df:DataFrame = inputCols
      .zip(topFeatures.toArray)
      .toList.toDF("Features", "Importance")
      .sort($"Importance".desc)

    println("Sorted input features in order of their importance:")
    df.show()
    println("End of decision tree model prediction and evaluation \n")
  }

  /** Trains a number of decision tree models using different hyper parameters.
    *
    * @param trainData
    * @param testData
    */

  def dTreeHyperTuned(trainData:DataFrame, testData:DataFrame) : Unit = {


    // Collecting input column names.
    val inputCols = trainData.columns.filter{colName=>
      colName.startsWith("Lag") &&
        !colName.contains("Date")}

    val pipeline = createDTreePipeline(inputCols)
    val decisionTree = pipeline.getStages.last.asInstanceOf[DecisionTreeClassifier]

  // Defining a search grid
  val parmGrid = new ParamGridBuilder()
    .addGrid(decisionTree.impurity, Seq("gini", "entropy"))
    .addGrid(decisionTree.maxDepth, Seq(1,2,3)) //,5,7,10)
    .addGrid(decisionTree.minInfoGain, Seq(0.01,0.02,0.04)) //
    .addGrid(decisionTree.minInstancesPerNode, Seq(20,50,100))
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

    println("Model evaluation metrics and their corresponding hyper parameter set \n" +
      "for top 10 best models sorted in order of their evaluation metrics:")
    validMetricAndHyperParam.take(10)
      .foreach{case (paramMap:ParamMap, valMetric:Double) =>
      println(s"Model accuracy is ${valMetric} for the following hyper paramers: \n ${paramMap}")}

  // Getting the best model (DecisionTreeClassificationModel) and its parameters
  val bestDtreeModel:DecisionTreeClassificationModel = tunedModel
    .bestModel
    .asInstanceOf[PipelineModel]
    .stages.last
    .asInstanceOf[DecisionTreeClassificationModel]

  // Most influential input features for the train decision tree model
  val topFeatures:linalg.Vector = bestDtreeModel
    .featureImportances

  val df:DataFrame = inputCols
    .zip(topFeatures.toArray)
    .toList.toDF("Features", "Importance")
    .sort($"Importance".desc)

    println("Sorted input features in order of their importance:")
    df.show()

  // Printing best model's (hyper-) parameters
    println("Best model's parameter are:")
  val parmMap = bestDtreeModel.extractParamMap()
    .toSeq.map{paramPair=>
    (paramPair.param.toString().split("__").apply(1), paramPair.value)}
    .foreach(println(_))
    println()

  // Printing the best fitted DecisionTree model structure
  println(bestDtreeModel.toDebugString)
  println("End of hyper parameter tuning for decision tree model!")
  }

  /** Creates a RandomForestClassifier pipeline using default hyper parameters.
    *
    * @param inputCols
    * @return
    */
  def createRndForestPipeline(inputCols: Array[String]):Pipeline = {

    val featureAndLabelStage = createFeatureAndLabelPipeline(inputCols)

    val randForest = new RandomForestClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("inputVector")
      .setLabelCol("WetDry")
      .setPredictionCol("Prediction")
      .setFeatureSubsetStrategy("all")
      .setSubsamplingRate(0.75)
      .setNumTrees(20)

    // Creating pipeline model and setting a RandomForest estimator:
    val pipeline = new Pipeline()
      .setStages(Array(featureAndLabelStage, randForest)) //first stage could be also a bucketizer

    pipeline
  }

  def rndForest(trainData:DataFrame, testData:DataFrame) : Unit = {
    // Collecting input column names.
    val inputCols = trainData.columns.filter{colName=>
      colName.startsWith("Lag") &&
        !colName.contains("Date")}

    // Creating a random forest pipeline
    val pipeline = createRndForestPipeline(inputCols)

    // Fitting the pipeline on trainData to create a model
    val pipelineModel = pipeline.fit(trainData)

    // Calculating model predictions for training and test data
    val trainDataPredictions = pipelineModel.transform(trainData)
    val testDataPredictions = pipelineModel.transform(testData)

    evaluateClassificationModel(dataFrame = trainDataPredictions,
      modelName = "Random Forest",
      dataFrameName = "trainData")

    evaluateClassificationModel(dataFrame = testDataPredictions,
      modelName = "Random Forest",
      dataFrameName = "testData")

    // Most influential input features for the random forest model
    val topFeatures:linalg.Vector = pipelineModel.stages.last
      .asInstanceOf[RandomForestClassificationModel].featureImportances

    val df:DataFrame = inputCols
      .zip(topFeatures.toArray)
      .toList.toDF("Features", "Importance")
      .sort($"Importance".desc)

    println("Sorted input features in order of their importance:")
    df.show()
    println("End of random forest model prediction and evaluation \n")
  }

  /** Fits a random forest model and tunes hyper parameters.
    *
    * @param trainData
    * @param testData
    */

  def rndForestHyperTuned(trainData: DataFrame, testData: DataFrame) : Unit = {

  // Collecting input column names.
  val inputCols = trainData.columns.filter{colName=>
    colName.startsWith("Lag") &&
      !colName.contains("Date")}

  val pipeline = createRndForestPipeline(inputCols)
  val rndForest = pipeline.getStages.last.asInstanceOf[RandomForestClassifier]

  // Defining a search grid
  val parmGrid = new ParamGridBuilder()
    .addGrid(rndForest.impurity, Seq("gini", "entropy"))
    .addGrid(rndForest.maxDepth, Seq(2,3))
    .addGrid(rndForest.minInfoGain, Seq(0.001, 0.005, 0.01, 0.02))
    .addGrid(rndForest.minInstancesPerNode, Seq(20,50,100))
    .addGrid(rndForest.numTrees, Seq(20,50))
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
    .setTrainRatio(0.85)

  // Training modelTuner
  val tunedModel = modelTuner.fit(trainData)

  // Printing combination of different hyper parameters and their evaluation metrics for validation data
  val validMetricAndHyperParam = tunedModel.getEstimatorParamMaps
    .zip(tunedModel.validationMetrics).sortBy(-_._2)

  println("Model evaluation metrics and their corresponding hyper parameter set \n" +
    "for top 10 best models sorted in order of their evaluation metrics:")
  validMetricAndHyperParam.take(10)
    .foreach{case (paramMap:ParamMap, valMetric:Double) =>
    println(s"Model accuracy is ${valMetric} for the following hyper paramers: \n ${paramMap}")}

  // Getting the best model (RandomForestClassificationModel) and its parameters
  val bestDtreeModel:RandomForestClassificationModel = tunedModel
    .bestModel
    .asInstanceOf[PipelineModel]
    .stages.last
    .asInstanceOf[RandomForestClassificationModel]

  // Most influential input features for the train decision tree model
  val topFeatures:linalg.Vector = bestDtreeModel
    .featureImportances

  val df:DataFrame = inputCols
    .zip(topFeatures.toArray)
    .toList.toDF("Features", "Importance")
    .sort($"Importance".desc)

  println("Sorted input features in order of their importance:")
  df.show()

  // Printing best model's (hyper-) parameters
  println("Best model's parameter are:")
  val parmMap = bestDtreeModel.extractParamMap()
    .toSeq.map{paramPair=>
    (paramPair.param.toString().split("__").apply(1), paramPair.value)}
    .foreach(println(_))
  println()

  // Printing the best fitted DecisionTree model structure
  println(bestDtreeModel.toDebugString)
  println("End of hyper parameter tuning for decision random forest classification model!")

  }

  def gbTree()







}




