package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.social_networks.IExercise10;
import uk.ac.cam.cl.mlrd.exercises.social_networks.IExercise12;

import java.util.*;
import java.util.stream.Collectors;

public class Exercise12 implements IExercise12 {
    private final double EPSILON = 1e-06;
    /**
     * Compute graph clustering using the Girvan-Newman method. Stop algorithm when the
     * minimum number of components has been reached (your answer may be higher than
     * the minimum).
     *
     * @param graph             {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *                          loaded graph
     * @param minimumComponents {@link int} The minimum number of components to reach.
     * @return {@link List}<{@link Set}<{@link Integer}>>
     * List of components for the graph.
     */
    @Override
    public List<Set<Integer>> GirvanNewman(Map<Integer, Set<Integer>> graph, int minimumComponents) {
        while (getComponents(graph).size() < minimumComponents && getNumberOfEdges(graph) > 0) {
            Map<Integer, Map<Integer, Double>> betweennesses = getEdgeBetweenness(graph);
            double max = betweennesses
                    .values()
                    .stream()
                    .flatMap(x -> x.values().stream())
                    .max(Double::compareTo)
                    .orElseThrow();
            Map<Integer, Set<Integer>> edgesToRemove = betweennesses
                    .entrySet()
                    .stream()
                    .collect(Collectors.toMap(Map.Entry::getKey,
                            x -> x.getValue().entrySet().stream()
                                    .filter(entry -> Math.abs(entry.getValue() - max) < EPSILON)
                                    .map(Map.Entry::getKey)
                                    .collect(Collectors.toSet())))
                    .entrySet()
                    .stream()
                    .filter(x -> !x.getValue().isEmpty())
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            for (int firstNode : edgesToRemove.keySet()) {
                for (int secondNode : edgesToRemove.get(firstNode)) {
                    graph.get(firstNode).remove(secondNode);
                }
            }
        }
        return getComponents(graph);
    }

    /**
     * Find the number of edges in the graph.
     *
     * @param graph {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *              loaded graph
     * @return {@link Integer}> Number of edges.
     */
    @Override
    public int getNumberOfEdges(Map<Integer, Set<Integer>> graph) {
        int total = 0;
        for (int vertex : graph.keySet()) {
            total += graph.get(vertex).size();
        }
        // undirected, not directed
        return total/2;
    }

    /**
     * Find the number of components in the graph using a DFS.
     *
     * @param graph {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *              loaded graph
     * @return {@link List}<{@link Set}<{@link Integer}>>
     * List of components for the graph.
     */
    @Override
    public List<Set<Integer>> getComponents(Map<Integer, Set<Integer>> graph) {
        Queue<Integer> unvisited = new ArrayDeque<>(graph.keySet());
        List<Set<Integer>> components = new ArrayList<>();
        while (!unvisited.isEmpty()) {
            int s = unvisited.poll();
            Set<Integer> component = new HashSet<>();
            visit(s, unvisited, graph, component);
            components.add(component);
        }
        return components;
    }

    private void visit(int vertex, Queue<Integer> unvisited, Map<Integer, Set<Integer>> graph, Set<Integer> currentComponent) {
        unvisited.remove(vertex);
        currentComponent.add(vertex);
        for (int neighbour : graph.get(vertex)) {
            if (unvisited.contains(neighbour)) {
                visit(neighbour, unvisited, graph, currentComponent);
            }
        }
    }

    /**
     * Calculate the edge betweenness.
     *
     * @param graph {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *              loaded graph
     * @return {@link Map}<{@link Integer},
     * {@link Map}<{@link Integer},{@link Double}>> Edge betweenness for
     * each pair of vertices in the graph
     */
    @Override
    public Map<Integer, Map<Integer, Double>> getEdgeBetweenness(Map<Integer, Set<Integer>> graph) {
        // overall data structure setup
        Map<Integer, Map<Integer, Double>> c_b = new HashMap<>();
        Queue<Integer> queue = new ArrayDeque<>();
        Stack<Integer> stack = new Stack<>();
        // initialise c_b values to 0
        for (int s : graph.keySet()) {
            c_b.put(s, new HashMap<>());
            for (int x : graph.keySet()) {
                c_b.get(s).put(x, 0.);
            }
        }
        // for each node:
        for (int s : graph.keySet()) {
            // initialise values for each node
            Map<Integer, List<Integer>> pred = new HashMap<>();
            Map<Integer, Integer> dist = new HashMap<>();
            Map<Integer, Integer> sigma = new HashMap<>();
            Map<Integer, Double> delta = new HashMap<>();
            for (int w : graph.keySet()) {
                pred.put(w, new ArrayList<>());
                dist.put(w, -1);
                sigma.put(w, 0);
                delta.put(w, 0.);
            }
            // special case for the current source
            dist.put(s, 0);
            sigma.put(s, 1);
            queue.add(s);
            // bfs
            while (!queue.isEmpty()) {
                int v = queue.poll();
                stack.add(v);
                // for each neighbour
                for (int w : graph.get(v)) {
                    // first time found?
                    if (dist.get(w) == -1) {
                        dist.put(w, dist.get(v) + 1);
                        queue.add(w);
                    }
                    // edge (v, w) on shortest path?
                    if (dist.get(w) == (dist.get(v) + 1)) {
                        sigma.put(w, sigma.get(w) + sigma.get(v));
                        pred.get(w).add(v);
                    }
                }
            }
            // back propagation of dependencies
            while (!stack.isEmpty()) {
                int w = stack.pop();
                for (int v : pred.get(w)) {
                    double c = ((double)sigma.get(v))/sigma.get(w) * (1 + delta.get(w));
                    c_b.get(v).put(w, c_b.get(v).get(w) + c);
                    delta.put(v, delta.get(v) + c);
                }
            }
        }
        List<Double> test = new ArrayList<>();
        for (int i : graph.keySet()) {
            for (int j : graph.keySet()) {
                c_b.get(i).put(j, c_b.get(i).get(j));
                test.add(c_b.get(i).get(j));
            }
        }
        return c_b;
    }
}
