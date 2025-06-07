package com.github.maximtereshchenko.snapdragon.api;

import java.util.List;

public record HiddenLayerConfiguration(List<Double> biases, List<List<Double>> weights) {}
