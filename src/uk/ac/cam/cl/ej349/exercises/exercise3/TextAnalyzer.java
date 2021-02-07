package uk.ac.cam.cl.ej349.exercises.exercise3;

import uk.ac.cam.cl.mlrd.exercises.sentiment_detection.Tokenizer;
import uk.ac.cam.cl.mlrd.utils.BestFit;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TextAnalyzer {

    private static List<Path> getReviews(Path reviewsDirectory) throws IOException {
        List<Path> reviewSet = new ArrayList<>();
        try (DirectoryStream<Path> reviews = Files.newDirectoryStream(reviewsDirectory)) {
            for (Path review : reviews) {
                reviewSet.add(review);
            }
        } catch (IOException e) {
            throw new IOException("Can't read the reviews at " + reviewsDirectory, e);
        }
        return reviewSet;
    }

    public static Map<String, Integer> getTypeFrequencies(Path reviewsDirectory) throws IOException {
        List<Path> reviewSet = getReviews(reviewsDirectory);
        Map<String, Integer> typeFrequencies = new HashMap<>();
        for (Path review : reviewSet) {
            try {
                List<String> tokens = Tokenizer.tokenize(review);
                for (String token : tokens) {
                    typeFrequencies.put(token, typeFrequencies.getOrDefault(token, 0) + 1);
                }
            } catch (IOException e) {
                throw new IOException("Can't read the review at " + review, e);
            }
        }
        return typeFrequencies;
    }

    public static Map<String, Integer> getTypeRanks(Map<String, Integer> typeFrequencies) {
        Map<String, Integer> typeRanks = new HashMap<>();
        List<Map.Entry<String, Integer>> sortedTypes = typeFrequencies.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).collect(Collectors.toList());
        int rank = 1;
        for(Map.Entry<String, Integer> type : sortedTypes) {
            typeRanks.put(type.getKey(), rank++);
        }
        return typeRanks;
    }

    public static List<BestFit.Point> numTypesAtTokens(Path reviewsDirectory) throws IOException {
        List<Path> reviewSet = getReviews(reviewsDirectory);
        Set<String> types = new TreeSet<>();
        List<BestFit.Point> points = new ArrayList<>();
        int numTokens = 0;
        for (Path review : reviewSet) {
            try {
                List<String> tokens = Tokenizer.tokenize(review);
                for (String token : tokens) {
                    types.add(token);
                    numTokens++;
                    double v = Math.log(numTokens) / Math.log(2);
                    if ((int)(Math.ceil(v)) == (int)(Math.floor(v))) {
                        points.add(new BestFit.Point(numTokens, types.size()));
                    }
                }
            } catch (IOException e) {
                throw new IOException("Can't read the review at " + review, e);
            }
        }
        points.add(new BestFit.Point(numTokens, types.size()));
        return points;
    }
}
