package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise1;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise2;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise5;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Sentiment;

import javax.print.attribute.standard.RequestingUserName;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class Exercise5 implements IExercise5 {
    @Override
    public List<Map<Path, Sentiment>> splitCVRandom(Map<Path, Sentiment> dataSet, int seed) {
        List<Map<Path, Sentiment>> foldsList = new ArrayList<>();
        List<Path> reviewSet = new ArrayList<>(dataSet.keySet());
        Collections.shuffle(reviewSet, new Random(seed));
        for (int i = 0; i < reviewSet.size(); i += reviewSet.size()/10) {
            List<Path> currentFoldKeys = reviewSet.subList(i, i + reviewSet.size()/10);
            Map<Path, Sentiment> currentFold = new HashMap<>();
            for (Path review : currentFoldKeys) {
                currentFold.put(review, dataSet.get(review));
            }
            foldsList.add(currentFold);
        }
        return foldsList;
    }

    @Override
    public List<Map<Path, Sentiment>> splitCVStratifiedRandom(Map<Path, Sentiment> dataSet, int seed) {
        List<Map<Path, Sentiment>> foldsList = new ArrayList<>();
        Map<Sentiment, List<Path>> reviewsBySentiment = new HashMap<>();
        for (Path review : dataSet.keySet()) {
            if (!reviewsBySentiment.containsKey(dataSet.get(review))) {
                reviewsBySentiment.put(dataSet.get(review), new ArrayList<>());
            }
            reviewsBySentiment.get(dataSet.get(review)).add(review);
        }
        for (Sentiment sentiment : reviewsBySentiment.keySet()) {
            Collections.shuffle(reviewsBySentiment.get(sentiment), new Random(seed));
        }
        for (int i = 0; i < dataSet.size()/2; i += dataSet.size()/20) {
            List<Path> positiveKeys = reviewsBySentiment.get(Sentiment.POSITIVE).subList(i, i + dataSet.size()/20);
            List<Path> negativeKeys = reviewsBySentiment.get(Sentiment.NEGATIVE).subList(i, i + dataSet.size()/20);
            List<Path> currentFoldKeys = new ArrayList<>();
            currentFoldKeys.addAll(positiveKeys);
            currentFoldKeys.addAll(negativeKeys);
            Map<Path, Sentiment> currentFold = new HashMap<>();
            for (Path review : currentFoldKeys) {
                currentFold.put(review, dataSet.get(review));
            }
            foldsList.add(currentFold);
        }
        return foldsList;
    }

    @Override
    public double[] crossValidate(List<Map<Path, Sentiment>> folds) throws IOException {
        List<Double> scores = new ArrayList<>();
        for (Map<Path, Sentiment> fold : folds) {
            Map<Path, Sentiment> trainingSet = new HashMap<>();
            for (Map<Path, Sentiment> fold1 : folds) {
                if (!fold1.equals(fold)) {
                    trainingSet.putAll(fold1);
                }
            }
            IExercise1 accuracyCalculator = (IExercise1) new Exercise1();
            IExercise2 implementation = (IExercise2) new Exercise2();
            Map<Sentiment, Double> classProbabilities =
                    implementation.calculateClassProbabilities(trainingSet);
            Map<String, Map<Sentiment, Double>> smoothedProbabilities =
                    implementation.calculateSmoothedLogProbs(trainingSet);
            Map<Path, Sentiment> predictedClassification =
                    implementation.naiveBayes(fold.keySet(), smoothedProbabilities, classProbabilities);
            double accuracy = accuracyCalculator.calculateAccuracy(fold, predictedClassification);
            scores.add(accuracy);
        }
        double[] toReturn = new double[scores.size()];
        for (int i = 0; i < scores.size(); i++) {
            toReturn[i] = scores.get(i);
        }
        return toReturn;
    }

    @Override
    public double cvAccuracy(double[] scores) {
        return Arrays.stream(scores).average().getAsDouble();
    }

    @Override
    public double cvVariance(double[] scores) {
        double mu = Arrays.stream(scores).average().getAsDouble();
        double sum = 0;
        for (double score : scores) {
            sum += Math.pow(score - mu, 2);
        }
        return sum / scores.length;
    }
}
