package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise4;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Sentiment;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Tokenizer;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.file.Path;
import java.util.*;

public class Exercise4 implements IExercise4 {
    @Override
    public Map<Path, Sentiment> magnitudeClassifier(Set<Path> testSet, Path lexiconFile) throws IOException {
        Exercise1 lexiconClassifier = new Exercise1();
        Map<String, Map<String, String>> lexiconData = lexiconClassifier.organiseLexicon(lexiconFile);
        Map<Path, Sentiment> reviewSentiments = new HashMap<>();
        for (Path review : testSet) {
            try {
                int score = 0;
                List<String> tokens = Tokenizer.tokenize(review);
                for (int i = 0; i < tokens.size(); i++) {
                    if (lexiconData.containsKey(tokens.get(i))) {
                        boolean positive = lexiconData.get(tokens.get(i)).get("Polarity").equals("positive");
                        boolean strong = lexiconData.get(tokens.get(i)).get("Intensity").equals("strong");
                        if (positive && strong) {
                            score+=2;
                        } else if (positive){
                            score++;
                        } else if (strong) {
                            score-=2;
                        } else {
                            score--;
                        }
                    }
                }
                Sentiment sentiment = score >= 0 ? Sentiment.POSITIVE : Sentiment.NEGATIVE;
                reviewSentiments.put(review, sentiment);
            } catch (IOException e) {
                throw new IOException("Could not find review at " + review, e);
            }
        }
        return reviewSentiments;
    }

    @Override
    public double signTest(Map<Path, Sentiment> actualSentiments, Map<Path, Sentiment> classificationA, Map<Path, Sentiment> classificationB) {
        int Plus = 0;
        int Minus = 0;
        int Null = 0;
        for (Path review : actualSentiments.keySet()) {
            Sentiment aPred = classificationA.get(review);
            Sentiment bPred = classificationB.get(review);
            Sentiment actual = actualSentiments.get(review);
            if (aPred == bPred) {
                Null += 1;
            } else if (aPred == actual) {
                Plus += 1;
            } else {
                Minus += 1;
            }
        }
        int n = (int) (2 * Math.ceil(Null / 2.) + Plus + Minus);
        int k = (int) (Math.ceil(Null / 2.) + Math.min(Plus, Minus));
        double pValue = 0;
        for (int i = 0; i <= k; i++) {
            pValue += new BigDecimal(choose(n, i)).multiply(BigDecimal.valueOf(Math.pow(0.5, i)))
                    .multiply(BigDecimal.valueOf(Math.pow(0.5, n - i))).doubleValue();
        }
        return pValue * 2;
    }

    public static void main(String[] args) {
        System.out.println(choose(230, 100));
    }

    private static BigInteger choose(int n, int k) {
        return factorial(BigInteger.valueOf(n)).divide(factorial(BigInteger.valueOf(k)))
                .divide(factorial(BigInteger.valueOf(n-k)));
    }

    private static BigInteger factorial(BigInteger integer) {
        BigInteger toReturn = BigInteger.ONE;
        while (integer.compareTo(BigInteger.ONE) > 0) {
            toReturn = toReturn.multiply(integer);
            integer = integer.subtract(BigInteger.ONE);
        }
        return toReturn;
    }
}
