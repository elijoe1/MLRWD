package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.markov_models.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class Exercise8 implements IExercise8 {
    @Override
    public List<DiceType> viterbi(HiddenMarkovModel<DiceRoll, DiceType> model, List<DiceRoll> observedSequence) {
        List<Map<DiceType, Double>> deltaTrellis = new ArrayList<>();
        List<Map<DiceType, DiceType>> phi = new ArrayList<>();

        Map<DiceType, Double> initialDelta = new HashMap<>();
        for (DiceType diceType : DiceType.values()) {
            initialDelta.put(diceType, Math.log(model.getEmissions(diceType).get(DiceRoll.START)));
        }

        deltaTrellis.add(initialDelta);

        for (int t = 1; t < observedSequence.size(); t++) {
            DiceRoll roll = observedSequence.get(t);
            Map<DiceType, Double> lastStateProbabilities = deltaTrellis.get(t-1);
            Map<DiceType, Double> deltaT = new HashMap<>();
            Map<DiceType, DiceType> phiT = new HashMap<>();
            for (DiceType diceType : DiceType.values()) {
                List<Double> probs = new ArrayList<>();
                double maxProb = 0.;
                DiceType stateForMaxProb = null;
                for (DiceType prevDiceType : DiceType.values()) {
                    double prob = lastStateProbabilities.get(prevDiceType) +
                            Math.log(model.getTransitions(prevDiceType).get(diceType)) +
                            Math.log(model.getEmissions(diceType).get(roll));
                    if (probs.stream().max(Double::compareTo).isEmpty() ||
                            probs.stream().max(Double::compareTo).get() < prob) {
                        maxProb = prob;
                        stateForMaxProb = prevDiceType;
                    }
                    probs.add(prob);
                }
                deltaT.put(diceType, maxProb);
                phiT.put(diceType, stateForMaxProb);
            }
            deltaTrellis.add(deltaT);
            phi.add(phiT);
        }

        List<DiceType> mostLikelySequence = new ArrayList<>();
        Collections.reverse(phi);
        mostLikelySequence.add(DiceType.END);
        int i = 0;
        DiceType next = phi.get(i).get(mostLikelySequence.get(i));
        while (next != DiceType.START) {
            mostLikelySequence.add(next);
            i++;
            next = phi.get(i).get(mostLikelySequence.get(i));
        }

        mostLikelySequence.add(DiceType.START);
        Collections.reverse(mostLikelySequence);

        return mostLikelySequence;
    }

    @Override
    public Map<List<DiceType>, List<DiceType>> predictAll(HiddenMarkovModel<DiceRoll, DiceType> model, List<Path> testFiles) throws IOException {
        List<HMMDataStore<DiceRoll, DiceType>> dataStore = HMMDataStore.loadDiceFiles(testFiles);
        Map<List<DiceType>, List<DiceType>> predictions = new HashMap<>();
        for (HMMDataStore<DiceRoll, DiceType> diceFile : dataStore) {
            predictions.put(diceFile.hiddenSequence, viterbi(model, diceFile.observedSequence));
        }
        return predictions;
    }

    @Override
    public double precision(Map<List<DiceType>, List<DiceType>> true2PredictedMap) {
        double predictedL = 0;
        double correctL = 0;
        for (Map.Entry<List<DiceType>, List<DiceType>> entry : true2PredictedMap.entrySet()) {
            List<DiceType> actualSequence = entry.getKey();
            List<DiceType> predictedSequence = entry.getValue();
            for (int i = 0; i < predictedSequence.size(); i++) {
                if (predictedSequence.get(i) == DiceType.WEIGHTED) {
                    predictedL++;
                    if (actualSequence.get(i) == DiceType.WEIGHTED) {
                        correctL++;
                    }
                }
            }
        }
        return correctL / predictedL;
    }

    @Override
    public double recall(Map<List<DiceType>, List<DiceType>> true2PredictedMap) {
        double actualL = 0;
        double correctL = 0;
        for (Map.Entry<List<DiceType>, List<DiceType>> entry : true2PredictedMap.entrySet()) {
            List<DiceType> actualSequence = entry.getKey();
            List<DiceType> predictedSequence = entry.getValue();
            for (int i = 0; i < predictedSequence.size(); i++) {
                if (actualSequence.get(i) == DiceType.WEIGHTED) {
                    actualL++;
                    if (predictedSequence.get(i) == DiceType.WEIGHTED) {
                        correctL++;
                    }
                }
            }
        }
        return correctL / actualL;
    }

    @Override
    public double fOneMeasure(Map<List<DiceType>, List<DiceType>> true2PredictedMap) {
        double precision = precision(true2PredictedMap);
        double recall = recall(true2PredictedMap);
        return 2 * (precision * recall) / (precision + recall);
    }
}
