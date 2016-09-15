/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016

Code Samples : Naive Bayes & Text PreProcessing

Problem Statement
*****************
The input data is a set of SMS messages that has been classified
as either "ham" or "spam". The goal of the exercise is to build a
 model to identify messages as either ham or spam.

## Techniques Used

1. Naive Bayes Classifier
2. Training and Testing
3. Confusion Matrix
4. Text Pre-Processing
5. Pipelines

*****************************************************************************/
package com.v2maestros.spark.bda.train;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;

public class SparkMLNaiveBayesDemo {

	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		final JavaSparkContext spContext = SparkConnection.getContext();
		final SparkSession spSession = SparkConnection.getSession();

		/*--------------------------------------------------------------------------
		Load Data
		--------------------------------------------------------------------------*/

		// Create the schema for the data to be loaded into Dataset.
		final StructType smsSchema = DataTypes
				.createStructType(new StructField[] { DataTypes.createStructField("label", DataTypes.DoubleType, false),
						DataTypes.createStructField("message", DataTypes.StringType, false) });

		final Dataset<Row> smsDf = spSession.read().csv("data/SMSSpamCollection.csv");
		smsDf.show(5);
		smsDf.printSchema();

		/*--------------------------------------------------------------------------
		Cleanse Data
		--------------------------------------------------------------------------*/
		// Cleanse Data and create a label/message dataframe.
		// Change data frame back to RDD
		final JavaRDD<Row> rdd1 = smsDf.toJavaRDD().repartition(2);

		// Function to map.
		final JavaRDD<Row> rdd2 = rdd1.map(new Function<Row, Row>() {

			@Override
			public Row call(Row iRow) throws Exception {

				final double spam = iRow.getString(0).equals("spam") ? 1.0 : 0.0;

				final Row retRow = RowFactory.create(spam, iRow.getString(1));

				return retRow;
			}

		});

		// Create Data Frame back.
		final Dataset<Row> smsCleansedDf = spSession.createDataFrame(rdd2, smsSchema);
		System.out.println("Transformed Data :");
		smsCleansedDf.show(5);

		/*--------------------------------------------------------------------------
		Prepare for Machine Learning - setup pipeline
		--------------------------------------------------------------------------*/
		// Split the data into training and test sets (30% held out for
		// testing).
		final Dataset<Row>[] splits = smsCleansedDf.randomSplit(new double[] { 0.7, 0.3 });
		final Dataset<Row> trainingData = splits[0];
		final Dataset<Row> testData = splits[1];

		/*--------------------------------------------------------------------------
		Perform machine learning. - Use the pipeline
		--------------------------------------------------------------------------*/
		final Tokenizer tokenizer = new Tokenizer().setInputCol("message").setOutputCol("words");
		final HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures");
		final IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");

		final NaiveBayes nbClassifier = new NaiveBayes().setLabelCol("label").setFeaturesCol("features");

		final Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] { tokenizer, hashingTF, idf, nbClassifier });

		final PipelineModel plModel = pipeline.fit(trainingData);

		final Dataset<Row> predictions = plModel.transform(testData);
		predictions.show(5);

		// View results
		System.out.println("Result sample :");
		predictions.show(5);

		// View confusion matrix
		System.out.println("Confusion Matrix :");
		predictions.groupBy(col("label"), col("prediction")).count().show();

		// Accuracy computation
		final MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy");
		final double accuracy = evaluator.evaluate(predictions);
		System.out.println("Accuracy = " + Math.round(accuracy * 100) + " %");

		// Keep the program running so we can checkout things.
		ExerciseUtils.hold();
	}

}
