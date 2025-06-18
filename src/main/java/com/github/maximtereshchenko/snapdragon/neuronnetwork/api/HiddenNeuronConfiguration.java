package com.github.maximtereshchenko.snapdragon.neuronnetwork.api;

import java.util.List;

public record HiddenNeuronConfiguration(double bias, List<Double> weights) {}
