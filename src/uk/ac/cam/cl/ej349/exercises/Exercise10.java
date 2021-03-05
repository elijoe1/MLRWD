package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.social_networks.IExercise10;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Exercise10 implements IExercise10 {

    /**
     * Load the graph file. Each line in the file corresponds to an edge; the
     * first column is the source node and the second column is the target. As
     * the graph is undirected, your program should add the source as a
     * neighbour of the target as well as the target a neighbour of the source.
     *
     * @param graphFile {@link Path} the path to the network specification
     * @return {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> For
     * each node, all the nodes neighbouring that node
     */
    @Override
    public Map<Integer, Set<Integer>> loadGraph(Path graphFile) throws IOException {
        Map<Integer, Set<Integer>> neighbours = new HashMap<>();
        BufferedReader reader = Files.newBufferedReader(graphFile);
        while (reader.ready()) {
            String line = reader.readLine();
            String[] nodes = line.split(" ");
            int source = Integer.parseInt(nodes[0]);
            int target = Integer.parseInt(nodes[1]);
            neighbours.putIfAbsent(source, new HashSet<>());
            neighbours.get(source).add(target);
            neighbours.putIfAbsent(target, new HashSet<>());
            neighbours.get(target).add(source);
        }
        return neighbours;
    }

    /**
     * Find the number of neighbours for each point in the graph.
     *
     * @param graph {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *              loaded graph
     * @return {@link Map}<{@link Integer}, {@link Integer}> For each node, the
     * number of neighbours it has
     */
    @Override
    public Map<Integer, Integer> getConnectivities(Map<Integer, Set<Integer>> graph) {
        Map<Integer, Integer> neighbourCount = new HashMap<>();
        for (Map.Entry<Integer, Set<Integer>> node : graph.entrySet()) {
            neighbourCount.putIfAbsent(node.getKey(), node.getValue().size());
        }
        return neighbourCount;
    }

    /**
     * Find the maximal shortest distance between any two nodes in the network
     * using a breadth-first search.
     *
     * @param graph {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *              loaded graph
     * @return <code>int</code> The diameter of the network
     */
    @Override
    public int getDiameter(Map<Integer, Set<Integer>> graph) {
        List<Integer> maxDistances = new ArrayList<>();
        for (Integer node : graph.keySet()) {
            maxDistances.add(longestShortestPath(graph, node));
        }
        return maxDistances.stream().max(Integer::compareTo).orElseThrow();
    }

    private int longestShortestPath(Map<Integer, Set<Integer>> graph, Integer root) {
        Queue<Integer> toExplore = new ArrayDeque<>();
        Map<Integer, Boolean> explored = new HashMap<>();
        Map<Integer, Integer> distances = new HashMap<>();
        for (Integer node : graph.keySet()) {
            explored.put(node, false);
        }
        toExplore.add(root);
        explored.put(root, true);
        distances.put(root, 0);
        while (!toExplore.isEmpty()) {
            int v = toExplore.poll();
            for (Integer neighbour : graph.get(v)) {
                if (!explored.get(neighbour)) {
                    toExplore.add(neighbour);
                    explored.put(neighbour, true);
                    distances.put(neighbour, distances.get(v) + 1);
                }
            }
        }
        return distances.values().stream().max(Integer::compareTo).orElseThrow();
    }
}
