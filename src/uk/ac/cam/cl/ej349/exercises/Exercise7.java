package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.markov_models.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class Exercise7 implements IExercise7 {
    @Override
    public HiddenMarkovModel<DiceRoll, DiceType> estimateHMM(Collection<Path> sequenceFiles) throws IOException {
        List<HMMDataStore<DiceRoll, DiceType>> dataStore = HMMDataStore.loadDiceFiles(sequenceFiles);

        Map<DiceType, Map<DiceType, Double>> transitionMatrix = new HashMap<>();
        for (DiceType currentType : DiceType.values()) {
            transitionMatrix.put(currentType, new HashMap<>());
        }

        Map<DiceType, Map<DiceRoll, Double>> emissionMatrix = new HashMap<>();
        for (DiceType diceType : DiceType.values()) {
            emissionMatrix.put(diceType, new HashMap<>());
        }

        for (HMMDataStore<DiceRoll, DiceType> data : dataStore) {
            List<DiceType> stateList = data.hiddenSequence;
            List<DiceRoll> observationList = data.observedSequence;
            Iterator<DiceType> states = stateList.iterator();
            Iterator<DiceRoll> observations = observationList.iterator();
            while (states.hasNext() && observations.hasNext()) {
                DiceType state = states.next();
                DiceRoll observation = observations.next();
                emissionMatrix.get(state).put(observation,
                        (emissionMatrix.get(state).getOrDefault(observation, 0.)) + 1.);
            }
        }

        for (HMMDataStore<DiceRoll, DiceType> data : dataStore) {
            List<DiceType> diceSequence = data.hiddenSequence;
            Iterator<DiceType> states = diceSequence.iterator();
            DiceType currentState = states.next();
            DiceType nextState;
            while (states.hasNext()) {
                nextState = states.next();
                for (DiceType currentType : DiceType.values()) {
                    for (DiceType nextType : DiceType.values()) {
                        if (currentState == currentType && nextState == nextType) {
                            transitionMatrix.get(currentType).put(nextType,
                                    transitionMatrix.get(currentType).getOrDefault(nextType, 0.) + 1.);
                        }
                    }
                }
                currentState = nextState;
            }
        }

        for (DiceType currentType : DiceType.values()) {
            double totalTransitions = transitionMatrix.get(currentType)
                    .values().stream().filter(Objects::nonNull).mapToDouble(Double::doubleValue).sum();
            for (DiceType nextType : DiceType.values()) {
                if (transitionMatrix.get(currentType).containsKey(nextType)) {
                    transitionMatrix.get(currentType).put(nextType,
                            (transitionMatrix.get(currentType).get(nextType)) / (totalTransitions));
                } else {
                    transitionMatrix.get(currentType).put(nextType, 0.);
                }
            }
        }

        for (DiceType state : DiceType.values()) {
            double totalObservations = emissionMatrix.get(state).values().stream()
                    .filter(Objects::nonNull).mapToDouble(Double::doubleValue).sum();
            for (DiceRoll observation : DiceRoll.values()) {
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

}
