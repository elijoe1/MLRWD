package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.IExercise1;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Sentiment;
import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Exercise1 implements IExercise1 {

    public Map<String, Map<String, String>> organiseLexicon(Path lexiconFile) throws IOException {
        Map<String, Map<String, String>> lexiconData = new HashMap<>();
        try {
            List<String> lexicon = Tokenizer.tokenize(lexiconFile);
            for (int i = 0; i < lexicon.size(); i+=9) {
                Map<String, String> wordData = new HashMap<>();
                wordData.put("Intensity", lexicon.get(i+5));
                wordData.put("Polarity", lexicon.get(i+8));
                lexiconData.put(lexicon.get(i+2), wordData);
            }
        } catch (IOException e) {
            throw new IOException("Lexicon file does not exist at " + lexiconFile, e);
        }
        return lexiconData;
    }

    @Override
    public Map<Path, Sentiment> simpleClassifier(Set<Path> testSet, Path lexiconFile) throws IOException {
        Map<String, Map<String, String>> lexiconData = organiseLexicon(lexiconFile);
        Map<Path, Sentiment> reviewSentiments = new HashMap<>();
        for (Path review : testSet) {
            try {
                int score = 0;
                List<String> tokens = Tokenizer.tokenize(review);
                for (String token : tokens) {
                    if (lexiconData.containsKey(token)) {
                        boolean positive = lexiconData.get(token).get("Polarity").equals("positive");
                        if (positive) {
                            score++;
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
    public double calculateAccuracy(Map<Path, Sentiment> trueSentiments, Map<Path, Sentiment> predictedSentiments) {
        double total = trueSentiments.size();
        double correct = 0.;
        for (Path key : trueSentiments.keySet()) {
            Sentiment predicted = predictedSentiments.get(key);
            Sentiment actual = trueSentiments.get(key);
            if (predicted == actual) {
                correct++;
            }
        }
        return correct/total;
    }

    @Override
    public Map<Path, Sentiment> improvedClassifier(Set<Path> testSet, Path lexiconFile) throws IOException {
        Map<String, Map<String, String>> lexiconData = organiseLexicon(lexiconFile);
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
                Sentiment sentiment = score >= 15 ? Sentiment.POSITIVE : Sentiment.NEGATIVE;
                reviewSentiments.put(review, sentiment);
            } catch (IOException e) {
                throw new IOException("Could not find review at " + review, e);
            }
        }
        return reviewSentiments;
    }
}
