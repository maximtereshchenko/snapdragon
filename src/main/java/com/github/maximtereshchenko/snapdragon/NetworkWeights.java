package com.github.maximtereshchenko.snapdragon;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

final class NetworkWeights {

    private final Map<AdjacentLayers, Weights> map;

    private NetworkWeights(Map<AdjacentLayers, Weights> map) {
        this.map = map;
    }

    NetworkWeights() {
        this(Map.of());
    }

    @Override
    public int hashCode() {
        return Objects.hash(map);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof NetworkWeights that &&
                   Objects.equals(map, that.map);
    }

    @Override
    public String toString() {
        return "NetworkWeights{" +
                   "map=" + map +
                   '}';
    }

    NetworkWeights with(LayerIndex left, LayerIndex right, Weights weights) {
        var adjacentLayers = new AdjacentLayers(left, right);
        if (map.containsKey(adjacentLayers)) {
            throw new IllegalArgumentException();
        }
        var copy = new HashMap<>(map);
        copy.put(adjacentLayers, weights);
        return new NetworkWeights(copy);
    }

    Weights weights(LayerIndex left, LayerIndex right) {
        return Objects.requireNonNull(map.get(new AdjacentLayers(left, right)));
    }

    NetworkWeights calibrated(
        LayerMap<Outputs> outputs,
        LayerMap<Deltas> deltas,
        LearningRate learningRate
    ) {
        return new NetworkWeights(
            map.entrySet()
                .stream()
                .collect(
                    Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> entry.getValue()
                                     .calibrated(
                                         outputs.element(entry.getKey().left()),
                                         deltas.element(entry.getKey().right()),
                                         learningRate
                                     )
                    )
                )
        );
    }

    private record AdjacentLayers(LayerIndex left, LayerIndex right) {}
}
