package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise2;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Sentiment;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Exercise2 implements IExercise2 {
    @Override
    public Map<Sentiment, Double> calculateClassProbabilities(Map<Path, Sentiment> trainingSet) throws IOException {
        double numPos = 0;
        double numNeg = 0;
        for (Path review : trainingSet.keySet()) {
            if (trainingSet.get(review) == Sentiment.POSITIVE) {
                numPos++;
            } else {
                numNeg++;
            }
        }
        Map<Sentiment, Double> classProbabilities = new HashMap<>();
        double posProb = numPos / (numPos + numNeg);
        double negProb = numNeg / (numPos + numNeg);
        classProbabilities.put(Sentiment.POSITIVE, posProb);
        classProbabilities.put(Sentiment.NEGATIVE, negProb);
        return classProbabilities;
    }

    @Override
    public Map<String, Map<Sentiment, Double>> calculateUnsmoothedLogProbs(Map<Path, Sentiment> trainingSet) throws IOException {
        Map<Sentiment, Double> totalTokenOccurrences = new HashMap<>();
        Map<String, Map<Sentiment, Double>> tokenOccurrences = new HashMap<>();
        Map<String, Map<Sentiment, Double>> tokenProbabilities = new HashMap<>();
        for (Path review : trainingSet.keySet()) {
            try {
                Sentiment actualSentiment = trainingSet.get(review);
                List<String> tokens = Tokenizer.tokenize(review);
                if (totalTokenOccurrences.containsKey(actualSentiment)) {
                    totalTokenOccurrences.put(actualSentiment, totalTokenOccurrences.get(actualSentiment) + tokens.size());
                } else {
                    totalTokenOccurrences.put(actualSentiment, (double) tokens.size());
                }
                for (String token : tokens) {
                    if (tokenOccurrences.containsKey(token) && tokenOccurrences.get(token).containsKey(actualSentiment)) {
                        tokenOccurrences.get(token).put(actualSentiment, tokenOccurrences.get(token).get(actualSentiment) + 1.);
                    } else if (tokenOccurrences.containsKey(token)){
                        tokenOccurrences.get(token).put(actualSentiment, 1.);
                    } else {
                        tokenOccurrences.put(token, new HashMap<>());
                        tokenOccurrences.get(token).put(actualSentiment, 1.);
                    }
                }
            } catch (IOException e) {
                throw new IOException("No review at " + review, e);
            }
        }
        for (String token : tokenOccurrences.keySet()) {
            Map<Sentiment, Double> probabilities = new HashMap<>();
            for (Sentiment sentiment : Sentiment.values()) {
                if (tokenOccurrences.get(token).get(sentiment) != null) {
                    probabilities.put(sentiment, Math.log(tokenOccurrences.get(token).get(sentiment) / totalTokenOccurrences.get(sentiment)));
                } else {
                    probabilities.put(sentiment, Math.log(0.));
                }
            }
            tokenProbabilities.put(token, probabilities);
        }
        return tokenProbabilities;
    }

    @Override
    public Map<String, Map<Sentiment, Double>> calculateSmoothedLogProbs(Map<Path, Sentiment> trainingSet) throws IOException {
        Map<Sentiment, Double> totalTokenOccurrences = new HashMap<>();
        Map<String, Map<Sentiment, Double>> tokenOccurrences = new HashMap<>();
        Map<String, Map<Sentiment, Double>> tokenProbabilities = new HashMap<>();
        for (Path review : trainingSet.keySet()) {
            try {
                Sentiment actualSentiment = trainingSet.get(review);
                List<String> tokens = Tokenizer.tokenize(review);
                if (totalTokenOccurrences.containsKey(actualSentiment)) {
                    totalTokenOccurrences.put(actualSentiment, totalTokenOccurrences.get(actualSentiment) + tokens.size());
                } else {
                    totalTokenOccurrences.put(actualSentiment, (double) tokens.size());
                }
                for (String token : tokens) {
                    if (tokenOccurrences.containsKey(token) && tokenOccurrences.get(token).containsKey(actualSentiment)) {
                        tokenOccurrences.get(token).put(actualSentiment, tokenOccurrences.get(token).get(actualSentiment) + 1.);
                    } else if (tokenOccurrences.containsKey(token)){
                        tokenOccurrences.get(token).put(actualSentiment, 1.);
                    } else {
                        tokenOccurrences.put(token, new HashMap<>());
                        tokenOccurrences.get(token).put(actualSentiment, 1.);
                    }
                }
            } catch (IOException e) {
                throw new IOException("No review at " + review, e);
            }
        }
        for (String token : tokenOccurrences.keySet()) {
            Map<Sentiment, Double> probabilities = new HashMap<>();
            for (Sentiment sentiment : Sentiment.values()) {
                if (tokenOccurrences.get(token).get(sentiment) != null) {
                    probabilities.put(sentiment, Math.log((tokenOccurrences.get(token).get(sentiment) + 1.) /
                            ((totalTokenOccurrences.get(sentiment)) + tokenOccurrences.size())));
                } else {
                    probabilities.put(sentiment, Math.log(1. / ((totalTokenOccurrences.get(sentiment)) + tokenOccurrences.size())));
                }
            }
            tokenProbabilities.put(token, probabilities);
        }
        return tokenProbabilities;
    }

    @Override
    public Map<Path, Sentiment> naiveBayes(Set<Path> testSet, Map<String, Map<Sentiment, Double>> tokenLogProbs, Map<Sentiment, Double> classProbabilities) throws IOException {
        Map<Path, Sentiment> predicted = new HashMap<>();
        for (Path review : testSet) {
            try {
                List<String> tokens = Tokenizer.tokenize(review);
                double posProb = Math.log(classProbabilities.get(Sentiment.POSITIVE));
                double negProb = Math.log(classProbabilities.get(Sentiment.POSITIVE));
                for (String token : tokens) {
                    if (tokenLogProbs.containsKey(token)) {
                        posProb += tokenLogProbs.get(token).get(Sentiment.POSITIVE);
                        negProb += tokenLogProbs.get(token).get(Sentiment.NEGATIVE);
                    }
                }
                predicted.put(review, (posProb >= negProb) ? Sentiment.POSITIVE : Sentiment.NEGATIVE);
            } catch (IOException e) {
                throw new IOException("No review at " + review, e);
            }
        }
        return predicted;
    }
}
