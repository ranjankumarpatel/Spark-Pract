package org.pract.spark.test;

import java.io.OutputStreamWriter;
import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

/**
 * Created by RANJAN on 9/15/2016.
 */
public class TestSNLP {

	public static void main(String[] args) {
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization,
		// NER, parsing, and coreference resolution
		final Properties props = new Properties();
		props.setProperty("annotators",
				"tokenize, ssplit, pos, lemma, ner, parse,depparse, dcoref,natlog,openie,sentiment");
		final StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		// read some text in the text variable
		final String text = "i don't know"; // Add your text here!

		// create an empty Annotation just with the given text
		final Annotation document = pipeline.process(text);

		// run all Annotators on this text
		pipeline.annotate(document);

		try {
			// pipeline.prettyPrint(document, System.out);
			pipeline.jsonPrint(document, new OutputStreamWriter(System.out));
		} catch (final Exception e) {
			e.printStackTrace();
		}

	}
}
