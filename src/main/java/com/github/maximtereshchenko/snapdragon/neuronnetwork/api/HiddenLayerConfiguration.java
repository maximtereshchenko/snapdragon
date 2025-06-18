package com.github.maximtereshchenko.snapdragon.neuronnetwork.api;

import java.util.List;

public record HiddenLayerConfiguration(
    List<HiddenNeuronConfiguration> hiddenNeuronConfigurations
) {}
