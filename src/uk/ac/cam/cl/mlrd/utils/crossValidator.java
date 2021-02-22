package uk.ac.cam.cl.mlrd.utils;

import uk.ac.cam.cl.ej349.exercises.*;
import uk.ac.cam.cl.mlrd.exercises.markov_models.*;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise1;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise2;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Sentiment;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class crossValidator {
    public static <T> List<List<T>> splitCVRandom(List<T> dataSet, int numFolds, int seed) {
        List<List<T>> foldsList = new ArrayList<>();
        Collections.shuffle(dataSet, new Random(seed));
        for (int i = 0; i < dataSet.size(); i += dataSet.size()/numFolds) {
            List<T> currentFold = dataSet.subList(i, i + dataSet.size()/numFolds);
            foldsList.add(currentFold);
        }
        return foldsList;
    }

    public static double cvVariance(double[] scores) {
        double mu = Arrays.stream(scores).average().getAsDouble();
        double sum = 0;
        for (double score : scores) {
            sum += Math.pow(score - mu, 2);
        }
        return sum / scores.length;
    }

    public static double cvAccuracy(double[] scores) {
        return Arrays.stream(scores).average().getAsDouble();
    }

    public static double[] crossValidate8(List<List<Path>> folds) throws IOException {
        List<Double> scores = new ArrayList<>();
        for (List<Path> fold : folds) {
            List<Path> trainingSet = new ArrayList<>();
            for (List<Path> fold1 : folds) {
                if (!fold1.equals(fold)) {
                    trainingSet.addAll(fold1);
                }
            }
            IExercise7 implementation7 = new Exercise7();
            IExercise8 implementation8 = new Exercise8();
            HiddenMarkovModel<DiceRoll, DiceType> model = implementation7.estimateHMM(trainingSet);
            Map<List<DiceType>, List<DiceType>> predictions = implementation8.predictAll(model, fold);
            scores.add(implementation8.fOneMeasure(predictions));
        }
        double[] toReturn = new double[scores.size()];
        for (int i = 0; i < scores.size(); i++) {
            toReturn[i] = scores.get(i);
        }
        return toReturn;
    }

    public static double[] crossValidate9(List<List<HMMDataStore<AminoAcid, Feature>>> folds) throws IOException {
        List<Double> scores = new ArrayList<>();
        for (List<HMMDataStore<AminoAcid, Feature>> fold : folds) {
            List<HMMDataStore<AminoAcid, Feature>> trainingSet = new ArrayList<>();
            for (List<HMMDataStore<AminoAcid, Feature>> fold1 : folds) {
                if (!fold1.equals(fold)) {
                    trainingSet.addAll(fold1);
                }
            }
            IExercise9 implementation = new Exercise9();
            HiddenMarkovModel<AminoAcid, Feature> model = implementation.estimateHMM(trainingSet);
            Map<List<Feature>, List<Feature>> predictions = implementation.predictAll(model, fold);
            scores.add(implementation.fOneMeasure(predictions));
        }
        double[] toReturn = new double[scores.size()];
        for (int i = 0; i < scores.size(); i++) {
            toReturn[i] = scores.get(i);
        }
        return toReturn;
    }
}
