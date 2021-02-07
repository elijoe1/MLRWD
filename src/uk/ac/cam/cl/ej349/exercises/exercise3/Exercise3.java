package uk.ac.cam.cl.ej349.exercises.exercise3;

import javafx.scene.chart.Chart;
import uk.ac.cam.cl.mlrd.utils.BestFit;
import uk.ac.cam.cl.mlrd.utils.ChartPlotter;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static uk.ac.cam.cl.ej349.exercises.exercise3.TextAnalyzer.*;
import static uk.ac.cam.cl.mlrd.utils.BestFit.leastSquares;

public class Exercise3 {

    private static List<BestFit.Point> top10000Points(Map<String, Integer> frequencies, Map<String, Integer> ranks) {
        List<BestFit.Point> points = new ArrayList<>();
        for (String type : ranks.keySet()) {
            if (ranks.get(type) <= 10000) {
                points.add(new BestFit.Point(ranks.get(type), frequencies.get(type)));
            }
        }
        return points;
    }

    private static List<BestFit.Point> my10Points(List<String> myTypes, Map<String, Integer> frequencies, Map<String, Integer> ranks) {
        List<BestFit.Point> points = new ArrayList<>();
        for (String type : myTypes) {
            points.add(new BestFit.Point(ranks.get(type), frequencies.get(type)));
        }
        return points;
    }

    private static List<BestFit.Point> logPoints(List<BestFit.Point> points) {
        List<BestFit.Point> logPoints = new ArrayList<>();
        for (BestFit.Point point : points) {
            logPoints.add(new BestFit.Point(Math.log(point.x), Math.log(point.y)));
        }
        return logPoints;
    }

    private static BestFit.Line zipfEstimate(List<BestFit.Point> points) {
        Map<BestFit.Point, Double> weightedPoints = new HashMap<>();
        for (BestFit.Point point : points) {
            weightedPoints.put(point, point.y);
        }
        return leastSquares(weightedPoints);
    }

    private static List<BestFit.Point> predictPoints(BestFit.Line bestFitLine, List<BestFit.Point> points) {
        List<BestFit.Point> predictedPoints = new ArrayList<>();
        for (BestFit.Point point : points) {
            predictedPoints.add(new BestFit.Point(point.x, bestFitLine.gradient * point.x + bestFitLine.yIntercept));
        }
        return predictedPoints;
    }

    private static Map<String, Map<String, Double>> evaluatePredictions(List<String> toPredict,
                                                                        BestFit.Line bestFit,
                                                                        Map<String, Integer> actualFreqs,
                                                                        Map<String, Integer> ranks) {
        Map<String, Map<String, Double>> evaluationData = new HashMap<>();
        for (String type : toPredict) {
            Double rank = (double) ranks.get(type);
            Double actualFreq = (double) actualFreqs.get(type);
            Double predictedFreq = Math.floor(Math.exp((Math.log(rank) * bestFit.gradient + bestFit.yIntercept)));
            evaluationData.put(type, new HashMap<String, Double>(Map.of("Predicted Freq.", predictedFreq, "Actual Freq.", actualFreq,
                    "% Difference", (predictedFreq/actualFreq) * 100 - 100)));
        }
        return evaluationData;
    }

    private static Map<String, Double> estimateParameters(BestFit.Line bestFitLine) {
        Map<String, Double> zipfParams = new HashMap<>();
        zipfParams.put("alpha", -1 * bestFitLine.gradient);
        zipfParams.put("k", Math.exp(bestFitLine.yIntercept));
        return zipfParams;
    }

    public static void main(String[] args) throws IOException {
        Path dataDirectory = Paths.get("data/large_dataset");
//        List<String> myLexicon = new ArrayList<>(List.of(
//                "best", "great", "tragic", "strange", "love", "mistaken", "nice", "pains", "profanity", "pretty"
//        ));
//        Map<String, Integer> typeFrequencies = getTypeFrequencies(dataDirectory);
//        Map<String, Integer> typeRanks = getTypeRanks(typeFrequencies);
//        List<BestFit.Point> top10000 = top10000Points(typeFrequencies, typeRanks);
//        List<BestFit.Point> my10 = my10Points(myLexicon, typeFrequencies, typeRanks);
//        BestFit.Line zipfEstimate = zipfEstimate(logPoints(top10000));
//        ChartPlotter.plotLines(predictPoints(zipfEstimate, logPoints(top10000)), logPoints(top10000));
//        System.out.println(evaluatePredictions(myLexicon, zipfEstimate, typeFrequencies, typeRanks));
//        System.out.println(estimateParameters(zipfEstimate));
//        System.out.println(zipfEstimate);
//        System.out.println(typeRanks.get("love"));
        ChartPlotter.plotLines(logPoints(numTypesAtTokens(dataDirectory)));
    }
}
