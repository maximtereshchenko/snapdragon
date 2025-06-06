package com.github.maximtereshchenko.snapdragon.api;

import java.util.List;

public record NeuralNetworkConfiguration(
    InputLayerConfiguration inputLayerConfiguration,
    List<HiddenLayerConfiguration> hiddenLayerConfigurations,
    OutputLayerConfiguration outputLayerConfiguration
) {}
