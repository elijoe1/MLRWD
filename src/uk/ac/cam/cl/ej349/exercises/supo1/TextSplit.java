package uk.ac.cam.cl.ej349.exercises.supo1;

import uk.ac.cam.cl.ej349.exercises.Exercise1;
import uk.ac.cam.cl.ej349.exercises.Exercise2;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.*;
import uk.ac.cam.cl.mlrd.utils.DataSplit;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class TextSplit {

    public static void main(String[] args) throws IOException {
//        Path review = Paths.get("data/supo1/supo1review");
//        try {
//            List<String> tokens = Tokenizer.tokenize(review);
//            Set<String> types = new HashSet<>(tokens);
//            List<String> typesList = new ArrayList<>(types);
//            Collections.sort(typesList);
//            for (String type : typesList) {
//                System.out.println(type);
//            }
//        } catch (IOException e) {
//            throw new IOException();
//        }

        Set<Path> reviews = new HashSet<>(Set.of(Paths.get("data/supo1/short_text_types")));
        Path lexicon = Paths.get("data/sentiment_lexicon");
        IExercise1 implementation = new Exercise1();
        System.out.println(implementation.simpleClassifier(reviews, lexicon));

        Path dataDirectory = Paths.get("data/sentiment_dataset");
        Path sentimentFile = dataDirectory.resolve("review_sentiment");
        Map<Path, Sentiment> dataSet = DataPreparation1.loadSentimentDataset(dataDirectory.resolve("reviews"),
                sentimentFile);
        DataSplit<Sentiment> split = new DataSplit<Sentiment>(dataSet, 0);

        IExercise2 implementation2 = new Exercise2();
        Map<Sentiment, Double> classProbs = implementation2.calculateClassProbabilities(split.trainingSet);
        Map<String, Map<Sentiment, Double>> typeProbs =  implementation2.calculateSmoothedLogProbs(split.trainingSet);
        System.out.println(implementation2.naiveBayes(reviews, typeProbs, classProbs));
    }
}
