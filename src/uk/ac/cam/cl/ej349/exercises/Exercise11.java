package uk.ac.cam.cl.ej349.exercises;

import uk.ac.cam.cl.mlrd.exercises.social_networks.IExercise10;
import uk.ac.cam.cl.mlrd.exercises.social_networks.IExercise11;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class Exercise11 implements IExercise11 {
    /**
     * Load the graph file. Use Brandes' algorithm to calculate the betweenness
     * centrality for each node in the graph.
     *
     * @param graphFile {@link Path} the path to the network specification
     * @return {@link Map}<{@link Integer}, {@link Double}> For
     * each node, its betweenness centrality
     */
    @Override
    public Map<Integer, Double> getNodeBetweenness(Path graphFile) throws IOException {
        // load graph
        IExercise10 graphLoader = new Exercise10();
        Map<Integer, Set<Integer>> graph = graphLoader.loadGraph(graphFile);
        // overall data structure setup
        Map<Integer, Double> c_b = new HashMap<>();
        Queue<Integer> queue = new ArrayDeque<>();
        Stack<Integer> stack = new Stack<>();
        // initialise c_b values to 0
        for (int s : graph.keySet()) {
            c_b.put(s, 0.);
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
                    delta.put(v, delta.get(v) + ((double)sigma.get(v))/sigma.get(w) * (1 + delta.get(w)));
                }
                if (w != s) {
                    c_b.put(w, c_b.get(w) + delta.get(w));
                }
            }
        }
        // halve all c_b values as the graph we have is given as undirected
        c_b.replaceAll((n, v) -> c_b.get(n) / 2.);
        return c_b;
    }
}
