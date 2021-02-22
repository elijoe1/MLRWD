package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.markov_models.*;

import java.io.IOException;
import java.util.*;

public class Exercise9 implements IExercise9 {
    @Override
    public HiddenMarkovModel<AminoAcid, Feature> estimateHMM(List<HMMDataStore<AminoAcid, Feature>> sequencePairs) throws IOException {

        Map<Feature, Map<Feature, Double>> transitionMatrix = new HashMap<>();
        for (Feature currentFeature : Feature.values()) {
            transitionMatrix.put(currentFeature, new HashMap<>());
        }

        Map<Feature, Map<AminoAcid, Double>> emissionMatrix = new HashMap<>();
        for (Feature feature : Feature.values()) {
            emissionMatrix.put(feature, new HashMap<>());
        }

        for (HMMDataStore<AminoAcid, Feature> data : sequencePairs) {
            List<Feature> stateList = data.hiddenSequence;
            List<AminoAcid> observationList = data.observedSequence;
            Iterator<Feature> states = stateList.iterator();
            Iterator<AminoAcid> observations = observationList.iterator();
            while (states.hasNext() && observations.hasNext()) {
                Feature state = states.next();
                AminoAcid observation = observations.next();
                emissionMatrix.get(state).put(observation,
                        (emissionMatrix.get(state).getOrDefault(observation, 0.)) + 1.);
            }
        }

        for (HMMDataStore<AminoAcid, Feature> data : sequencePairs) {
            List<Feature> stateSequence = data.hiddenSequence;
            Iterator<Feature> states = stateSequence.iterator();
            Feature currentState = states.next();
            Feature nextState;
            while (states.hasNext()) {
                nextState = states.next();
                for (Feature currentType : Feature.values()) {
                    for (Feature nextType : Feature.values()) {
                        if (currentState == currentType && nextState == nextType) {
                            transitionMatrix.get(currentType).put(nextType,
                                    transitionMatrix.get(currentType).getOrDefault(nextType, 0.) + 1.);
                        }
                    }
                }
                currentState = nextState;
            }
        }

        for (Feature currentType : Feature.values()) {
            double totalTransitions = transitionMatrix.get(currentType)
                    .values().stream().filter(Objects::nonNull).mapToDouble(Double::doubleValue).sum();
            for (Feature nextType : Feature.values()) {
                if (transitionMatrix.get(currentType).containsKey(nextType)) {
                    transitionMatrix.get(currentType).put(nextType,
                            (transitionMatrix.get(currentType).get(nextType)) / (totalTransitions));
                } else {
                    transitionMatrix.get(currentType).put(nextType, 0.);
                }
            }
        }

        for (Feature state : Feature.values()) {
            double totalObservations = emissionMatrix.get(state).values().stream()
                    .filter(Objects::nonNull).mapToDouble(Double::doubleValue).sum();
            for (AminoAcid observation : AminoAcid.values()) {
                if (emissionMatrix.get(state).containsKey(observation)) {
                    emissionMatrix.get(state).put(observation,
                            (emissionMatrix.get(state).get(observation)) / (totalObservations));
                } else {
                    emissionMatrix.get(state).put(observation, 0.);
                }
            }
        }

        return new HiddenMarkovModel<>(transitionMatrix, emissionMatrix);
    }

    @Override
    public List<Feature> viterbi(HiddenMarkovModel<AminoAcid, Feature> model, List<AminoAcid> observedSequence) {
        List<Map<Feature, Double>> deltaTrellis = new ArrayList<>();
        List<Map<Feature, Feature>> phi = new ArrayList<>();

        Map<Feature, Double> initialDelta = new HashMap<>();
        for (Feature feature : Feature.values()) {
            initialDelta.put(feature, Math.log(model.getEmissions(feature).get(AminoAcid.START)));
        }

        deltaTrellis.add(initialDelta);

        for (int t = 1; t < observedSequence.size(); t++) {
            AminoAcid acid = observedSequence.get(t);
            Map<Feature, Double> lastStateProbabilities = deltaTrellis.get(t-1);
            Map<Feature, Double> deltaT = new HashMap<>();
            Map<Feature, Feature> phiT = new HashMap<>();
            for (Feature feature : Feature.values()) {
                List<Double> probs = new ArrayList<>();
                double maxProb = 0.;
                Feature stateForMaxProb = null;
                for (Feature prevFeature : Feature.values()) {
                    double prob = lastStateProbabilities.get(prevFeature) +
                            Math.log(model.getTransitions(prevFeature).get(feature)) +
                            Math.log(model.getEmissions(feature).get(acid));
                    if (probs.stream().max(Double::compareTo).isEmpty() ||
                            probs.stream().max(Double::compareTo).get() < prob) {
                        maxProb = prob;
                        stateForMaxProb = prevFeature;
                    }
                    probs.add(prob);
                }
                deltaT.put(feature, maxProb);
                phiT.put(feature, stateForMaxProb);
            }
            deltaTrellis.add(deltaT);
            phi.add(phiT);
        }

        List<Feature> mostLikelySequence = new ArrayList<>();
        Collections.reverse(phi);
        mostLikelySequence.add(Feature.END);
        int i = 0;
        Feature next = phi.get(i).get(mostLikelySequence.get(i));
        while (next != Feature.START) {
            mostLikelySequence.add(next);
            i++;
            next = phi.get(i).get(mostLikelySequence.get(i));
        }

        mostLikelySequence.add(Feature.START);
        Collections.reverse(mostLikelySequence);

        return mostLikelySequence;
    }

    @Override
    public Map<List<Feature>, List<Feature>> predictAll(HiddenMarkovModel<AminoAcid, Feature> model, List<HMMDataStore<AminoAcid, Feature>> testSequencePairs) throws IOException {
        Map<List<Feature>, List<Feature>> predictions = new HashMap<>();
        for (HMMDataStore<AminoAcid, Feature> sequencePair : testSequencePairs) {
            predictions.put(sequencePair.hiddenSequence, viterbi(model, sequencePair.observedSequence));
        }
        return predictions;
    }

    @Override
    public double precision(Map<List<Feature>, List<Feature>> true2PredictedMap) {
        double predicted = 0;
        double correct = 0;
        for (Map.Entry<List<Feature>, List<Feature>> entry : true2PredictedMap.entrySet()) {
            List<Feature> actualSequence = entry.getKey();
            List<Feature> predictedSequence = entry.getValue();
            for (int i = 0; i < predictedSequence.size(); i++) {
                if (predictedSequence.get(i) == Feature.MEMBRANE) {
                    predicted++;
                    if (actualSequence.get(i) == Feature.MEMBRANE) {
                        correct++;
                    }
                }
            }
        }
        return correct / predicted;
    }

    @Override
    public double recall(Map<List<Feature>, List<Feature>> true2PredictedMap) {
        double actual = 0;
        double correct = 0;
        for (Map.Entry<List<Feature>, List<Feature>> entry : true2PredictedMap.entrySet()) {
            List<Feature> actualSequence = entry.getKey();
            List<Feature> predictedSequence = entry.getValue();
            for (int i = 0; i < predictedSequence.size(); i++) {
                if (actualSequence.get(i) == Feature.MEMBRANE) {
                    actual++;
                    if (predictedSequence.get(i) == Feature.MEMBRANE) {
                        correct++;
                    }
                }
            }
        }
        return correct / actual;
    }

    @Override
    public double fOneMeasure(Map<List<Feature>, List<Feature>> true2PredictedMap) {
        double precision = precision(true2PredictedMap);
        double recall = recall(true2PredictedMap);
        return 2 * (precision * recall) / (precision + recall);
    }
}
