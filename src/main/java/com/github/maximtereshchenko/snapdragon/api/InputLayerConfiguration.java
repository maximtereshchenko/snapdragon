package com.github.maximtereshchenko.snapdragon.api;

import java.util.List;

public record InputLayerConfiguration(
    List<InputNeuronConfiguration> inputNeuronConfigurations
) {}
