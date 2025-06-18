package com.github.maximtereshchenko.snapdragon.neuronnetwork.api;

import java.util.List;

public record OutputLayerConfiguration(
    List<OutputNeuronConfiguration> outputNeuronConfigurations
) {}
