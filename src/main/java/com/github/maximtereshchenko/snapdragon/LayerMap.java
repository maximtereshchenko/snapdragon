package com.github.maximtereshchenko.snapdragon;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

final class LayerMap<T> {

    private final Map<LayerIndex, T> map;

    private LayerMap(Map<LayerIndex, T> map) {
        this.map = map;
    }

    LayerMap(LayerIndex layerIndex, T element) {
        this(Map.of(layerIndex, element));
    }

    LayerMap<T> with(LayerIndex layerIndex, T element) {
        if (map.containsKey(layerIndex)) {
            throw new IllegalArgumentException();
        }
        var copy = new HashMap<>(map);
        copy.put(layerIndex, element);
        return new LayerMap<>(copy);
    }

    T element(LayerIndex layerIndex) {
        return Objects.requireNonNull(map.get(layerIndex));
    }
}
