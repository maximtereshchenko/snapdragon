package com.github.maximtereshchenko.snapdragon.api;

import java.util.List;

public record InputLayerConfiguration(int inputs, List<List<Double>> weights) {}
