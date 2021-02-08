package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class Exercise6 implements IExercise6 {
    @Override
    public Map<NuancedSentiment, Double> calculateClassProbabilities(Map<Path, NuancedSentiment> trainingSet) throws IOException {
        double numPos = 0;
        double numNeg = 0;
        double numNeut = 0;
        for (Path review : trainingSet.keySet()) {
            if (trainingSet.get(review) == NuancedSentiment.POSITIVE) {
                numPos++;
            } else if (trainingSet.get(review) == NuancedSentiment.NEGATIVE) {
                numNeg++;
            } else {
                numNeut++;
            }
        }
        Map<NuancedSentiment, Double> classProbabilities = new HashMap<>();
        double posProb = numPos / (numPos + numNeg + numNeut);
        double negProb = numNeg / (numPos + numNeg + numNeut);
        double neutProb = numNeut / (numPos + numNeg + numNeut);
        classProbabilities.put(NuancedSentiment.POSITIVE, posProb);
        classProbabilities.put(NuancedSentiment.NEGATIVE, negProb);
        classProbabilities.put(NuancedSentiment.NEUTRAL, neutProb);
        return classProbabilities;
    }

    @Override
    public Map<String, Map<NuancedSentiment, Double>> calculateNuancedLogProbs(Map<Path, NuancedSentiment> trainingSet) throws IOException {
        Map<NuancedSentiment, Double> totalTokenOccurrences = new HashMap<>();
        Map<String, Map<NuancedSentiment, Double>> tokenOccurrences = new HashMap<>();
        Map<String, Map<NuancedSentiment, Double>> tokenProbabilities = new HashMap<>();
        for (Path review : trainingSet.keySet()) {
            try {
                NuancedSentiment actualSentiment = trainingSet.get(review);
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
            Map<NuancedSentiment, Double> probabilities = new HashMap<>();
            for (NuancedSentiment sentiment : NuancedSentiment.values()) {
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
    public Map<Path, NuancedSentiment> nuancedClassifier(Set<Path> testSet, Map<String, Map<NuancedSentiment, Double>> tokenLogProbs, Map<NuancedSentiment, Double> classProbabilities) throws IOException {
        Map<Path, NuancedSentiment> predicted = new HashMap<>();
        for (Path review : testSet) {
            try {
                List<String> tokens = Tokenizer.tokenize(review);
                double posProb = Math.log(classProbabilities.get(NuancedSentiment.POSITIVE));
                double negProb = Math.log(classProbabilities.get(NuancedSentiment.NEGATIVE));
                double neutProb = Math.log(classProbabilities.get(NuancedSentiment.NEUTRAL));
                for (String token : tokens) {
                    if (tokenLogProbs.containsKey(token)) {
                        posProb += tokenLogProbs.get(token).get(NuancedSentiment.POSITIVE);
                        negProb += tokenLogProbs.get(token).get(NuancedSentiment.NEGATIVE);
                        neutProb += tokenLogProbs.get(token).get(NuancedSentiment.NEUTRAL);
                    }
                }
                if (posProb >= neutProb) {
                    if (posProb >= negProb) {
                        predicted.put(review, NuancedSentiment.POSITIVE);
                    } else {
                        predicted.put(review, NuancedSentiment.NEGATIVE);
                    }
                } else if (neutProb >= negProb) {
                    predicted.put(review, NuancedSentiment.NEUTRAL);
                } else {
                    predicted.put(review, NuancedSentiment.NEGATIVE);
                }
            } catch (IOException e) {
                throw new IOException("No review at " + review, e);
            }
        }
        return predicted;
    }

    @Override
    public double nuancedAccuracy(Map<Path, NuancedSentiment> trueSentiments, Map<Path, NuancedSentiment> predictedSentiments) {
        double total = trueSentiments.size();
        double correct = 0.;
        for (Path key : trueSentiments.keySet()) {
            NuancedSentiment predicted = predictedSentiments.get(key);
            NuancedSentiment actual = trueSentiments.get(key);
            if (predicted == actual) {
                correct++;
            }
        }
        return correct/total;
    }

    @Override
    public Map<Integer, Map<Sentiment, Integer>> agreementTable(Collection<Map<Integer, Sentiment>> predictedSentiments) {
        Map<Integer, Map<Sentiment, Integer>> table = new HashMap<>();
        for (Map<Integer, Sentiment> predictionSet : predictedSentiments) {
            for (Integer reviewNumber : predictionSet.keySet()) {
                if (!table.containsKey(reviewNumber)) {
                    table.put(reviewNumber, new HashMap<>());
                }
                table.get(reviewNumber).put(predictionSet.get(reviewNumber), table.get(reviewNumber)
                        .getOrDefault(predictionSet.get(reviewNumber), 0) + 1);
            }
        }
        return table;
    }

    public double chanceAgreement(Map<Integer, Map<Sentiment, Integer>> agreementTable) {
        double agreement = 0;
        for (Sentiment sentiment : Sentiment.values()) {
            double classCount = 0;
            double numJudges = 0;
            for (Integer review : agreementTable.keySet()) {
                double nij = agreementTable.get(review).getOrDefault(sentiment,0);
                classCount += nij;
                numJudges = agreementTable.get(review).values().stream().mapToInt(Integer::intValue).sum();
            }
            double classProb = classCount / (agreementTable.size() * numJudges);
            agreement += Math.pow(classProb, 2);
        }
        return agreement;
    }

    public double observedAgreement(Map<Integer, Map<Sentiment, Integer>> agreementTable) {
        double agreement = 0;
        for (Integer review : agreementTable.keySet()) {
            double observedSum = 0;
            for (Sentiment sentiment : Sentiment.values()) {
                double nij = agreementTable.get(review).getOrDefault(sentiment,0);
                observedSum += ((nij) * (nij - 1));
            }
            double numJudges = agreementTable.get(review).values().stream().mapToInt(Integer::intValue).sum();
            double totalPossible = (numJudges * (numJudges - 1));
            double si = observedSum / totalPossible;
            agreement += si;
        }
        return agreement / agreementTable.size();
    }

    @Override
    public double kappa(Map<Integer, Map<Sentiment, Integer>> agreementTable) {
        double pe = chanceAgreement(agreementTable);
        double pa = observedAgreement(agreementTable);
        double kappa = (pa - pe) / (1 - pe);
        return kappa;
    }

    // Modified cross validation methods to account for more NuancedSentiment instead of Sentiment
    public List<Map<Path, NuancedSentiment>> nuancedSplitCVStratifiedRandom(Map<Path, NuancedSentiment> dataSet, int seed) {
        List<Map<Path, NuancedSentiment>> foldsList = new ArrayList<>();
        Map<NuancedSentiment, List<Path>> reviewsBySentiment = new HashMap<>();
        for (Path review : dataSet.keySet()) {
            if (!reviewsBySentiment.containsKey(dataSet.get(review))) {
                reviewsBySentiment.put(dataSet.get(review), new ArrayList<>());
            }
            reviewsBySentiment.get(dataSet.get(review)).add(review);
        }
        for (NuancedSentiment sentiment : reviewsBySentiment.keySet()) {
            Collections.shuffle(reviewsBySentiment.get(sentiment), new Random(seed));
        }
        for (int i = 0; i < dataSet.size()/3; i += dataSet.size()/30) {
            List<Path> positiveKeys = reviewsBySentiment.get(NuancedSentiment.POSITIVE).subList(i, i + dataSet.size()/30);
            List<Path> negativeKeys = reviewsBySentiment.get(NuancedSentiment.NEGATIVE).subList(i, i + dataSet.size()/30);
            List<Path> neutralKeys = reviewsBySentiment.get(NuancedSentiment.NEUTRAL).subList(i, i + dataSet.size()/30);
            List<Path> currentFoldKeys = new ArrayList<>();
            currentFoldKeys.addAll(positiveKeys);
            currentFoldKeys.addAll(negativeKeys);
            currentFoldKeys.addAll(neutralKeys);
            Map<Path, NuancedSentiment> currentFold = new HashMap<>();
            for (Path review : currentFoldKeys) {
                currentFold.put(review, dataSet.get(review));
            }
            foldsList.add(currentFold);
        }
        return foldsList;
    }

    public double[] nuancedCrossValidate(List<Map<Path, NuancedSentiment>> folds) throws IOException {
        List<Double> scores = new ArrayList<>();
        for (Map<Path, NuancedSentiment> fold : folds) {
            Map<Path, NuancedSentiment> trainingSet = new HashMap<>();
            for (Map<Path, NuancedSentiment> fold1 : folds) {
                if (!fold1.equals(fold)) {
                    trainingSet.putAll(fold1);
                }
            }
            IExercise6 implementation = (IExercise6) new Exercise6();
            Map<NuancedSentiment, Double> classProbabilities =
                    implementation.calculateClassProbabilities(trainingSet);
            Map<String, Map<NuancedSentiment, Double>> smoothedProbabilities =
                    implementation.calculateNuancedLogProbs(trainingSet);
            Map<Path, NuancedSentiment> predictedClassification =
                    implementation.nuancedClassifier(fold.keySet(), smoothedProbabilities, classProbabilities);
            double accuracy = implementation.nuancedAccuracy(fold, predictedClassification);
            scores.add(accuracy);
        }
        double[] toReturn = new double[scores.size()];
        for (int i = 0; i < scores.size(); i++) {
            toReturn[i] = scores.get(i);
        }
        return toReturn;
    }

}
