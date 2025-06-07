package com.github.maximtereshchenko.snapdragon.api;

import java.util.List;

public record OutputLayerConfiguration(
    List<OutputNeuronConfiguration> outputNeuronConfigurations
) {}
