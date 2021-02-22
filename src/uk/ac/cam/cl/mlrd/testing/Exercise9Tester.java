package uk.ac.cam.cl.mlrd.testing;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//TODO: Replace with your package.
import uk.ac.cam.cl.ej349.exercises.Exercise9;
import uk.ac.cam.cl.mlrd.exercises.markov_models.AminoAcid;
import uk.ac.cam.cl.mlrd.exercises.markov_models.Feature;
import uk.ac.cam.cl.mlrd.exercises.markov_models.HMMDataStore;
import uk.ac.cam.cl.mlrd.exercises.markov_models.HiddenMarkovModel;
import uk.ac.cam.cl.mlrd.exercises.markov_models.IExercise9;
import uk.ac.cam.cl.mlrd.utils.crossValidator;

public class Exercise9Tester {

	static final Path dataFile = Paths.get("data/bio_dataset.txt");

	public static void main(String[] args) throws IOException {

		List<HMMDataStore<AminoAcid, Feature>> sequencePairs = HMMDataStore.loadBioFile(dataFile);

		System.out.println(sequencePairs.size());

		// Use for testing the code
		Collections.shuffle(sequencePairs, new Random(0));
		int testSize = sequencePairs.size() / 10;
		List<HMMDataStore<AminoAcid, Feature>> devSet = sequencePairs.subList(0, testSize);
		List<HMMDataStore<AminoAcid, Feature>> testSet = sequencePairs.subList(testSize, 2 * testSize);
		List<HMMDataStore<AminoAcid, Feature>> trainingSet = sequencePairs.subList(testSize * 2, sequencePairs.size());
		// But:
		// TODO: Replace with cross-validation for the tick.
		List<List<HMMDataStore<AminoAcid, Feature>>> foldsList = crossValidator.splitCVRandom(sequencePairs, 10, 0);
		double[] FMeasures = crossValidator.crossValidate9(foldsList);
		double meanFMeasure = crossValidator.cvAccuracy(FMeasures);
		double varianceFMeasure = crossValidator.cvVariance(FMeasures);
		System.out.println("F measures for each fold:");
		System.out.println(Arrays.toString(FMeasures));
		System.out.println();
		System.out.println("Average F measure with cross validation:");
		System.out.println(meanFMeasure);
		System.out.println();
		System.out.println("F measure variance with cross validation");
		System.out.println(varianceFMeasure);
		System.out.println();

		IExercise9 implementation = (IExercise9) new Exercise9();

		HiddenMarkovModel<AminoAcid, Feature> model = implementation.estimateHMM(trainingSet);
		System.out.println("Predicted transitions:");
		System.out.println(model.getTransitionMatrix());
		System.out.println();
		System.out.println("Predicted emissions:");
		System.out.println(model.getEmissionMatrix());
		System.out.println();

		HMMDataStore<AminoAcid, Feature> data = devSet.get(0);
		List<Feature> predicted = implementation.viterbi(model, data.observedSequence);
		System.out.println("True hidden sequence:");
		System.out.println(data.hiddenSequence);
		System.out.println();

		System.out.println("Predicted hidden sequence:");
		System.out.println(predicted);
		System.out.println();

		Map<List<Feature>, List<Feature>> true2PredictedSequences = implementation.predictAll(model, devSet);
		double accuracy = implementation.precision(true2PredictedSequences);
		System.out.println("Prediction precision:");
		System.out.println(accuracy);
		System.out.println();

		double recall = implementation.recall(true2PredictedSequences);
		System.out.println("Prediction recall:");
		System.out.println(recall);
		System.out.println();

		double f1Score = implementation.fOneMeasure(true2PredictedSequences);
		System.out.println("Prediction F1 score:");
		System.out.println(f1Score);
		System.out.println();
	}
}
