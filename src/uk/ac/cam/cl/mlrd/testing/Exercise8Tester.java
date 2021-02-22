package uk.ac.cam.cl.mlrd.testing;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

//TODO: Replace with your package.
import uk.ac.cam.cl.ej349.exercises.Exercise7;
import uk.ac.cam.cl.ej349.exercises.Exercise8;
import uk.ac.cam.cl.mlrd.exercises.markov_models.DiceRoll;
import uk.ac.cam.cl.mlrd.exercises.markov_models.DiceType;
import uk.ac.cam.cl.mlrd.exercises.markov_models.HMMDataStore;
import uk.ac.cam.cl.mlrd.exercises.markov_models.HiddenMarkovModel;
import uk.ac.cam.cl.mlrd.exercises.markov_models.IExercise7;
import uk.ac.cam.cl.mlrd.exercises.markov_models.IExercise8;
import uk.ac.cam.cl.mlrd.utils.crossValidator;

public class Exercise8Tester {

	static final Path dataDirectory = Paths.get("data/dice_dataset");

	public static void main(String[] args)
			throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException {

		List<Path> sequenceFiles = new ArrayList<>();
		try (DirectoryStream<Path> files = Files.newDirectoryStream(dataDirectory)) {
			for (Path item : files) {
				sequenceFiles.add(item);
			}
		} catch (IOException e) {
			throw new IOException("Cant access the dataset.", e);
		}

		// Use for testing the code
		Collections.shuffle(sequenceFiles, new Random(0));
		int testSize = sequenceFiles.size() / 10;
		List<Path> devSet = sequenceFiles.subList(0, testSize);
		List<Path> testSet = sequenceFiles.subList(testSize, 2 * testSize);
		List<Path> trainingSet = sequenceFiles.subList(testSize * 2, sequenceFiles.size());
		// But:
		// TODO: Replace with cross-validation for the tick.
		List<List<Path>> foldsList = crossValidator.splitCVRandom(sequenceFiles, 10, 0);
		double[] FMeasures = crossValidator.crossValidate8(foldsList);
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

		IExercise7 implementation7 = (IExercise7) new Exercise7();
		HiddenMarkovModel<DiceRoll, DiceType> model = implementation7.estimateHMM(trainingSet);

		IExercise8 implementation = (IExercise8) new Exercise8();

		HMMDataStore<DiceRoll, DiceType> data = HMMDataStore.loadDiceFile(devSet.get(0));
		List<DiceType> predicted = implementation.viterbi(model, data.observedSequence);
		System.out.println("True hidden sequence:");
		System.out.println(data.hiddenSequence);
		System.out.println();

		System.out.println("Predicted hidden sequence:");
		System.out.println(predicted);
		System.out.println();

		Map<List<DiceType>, List<DiceType>> true2PredictedMap = implementation.predictAll(model, devSet);
		double precision = implementation.precision(true2PredictedMap);
		System.out.println("Prediction precision:");
		System.out.println(precision);
		System.out.println();

		double recall = implementation.recall(true2PredictedMap);
		System.out.println("Prediction recall:");
		System.out.println(recall);
		System.out.println();

		double fOneMeasure = implementation.fOneMeasure(true2PredictedMap);
		System.out.println("Prediction fOneMeasure:");
		System.out.println(fOneMeasure);
		System.out.println();
	}
}
