package com.github.maximtereshchenko.snapdragon;

record Biases(Tensor tensor) {

    Biases calibrated(Deltas deltas, LearningRate learningRate) {
        var deltasTensor = deltas.tensor();
        var shape = deltasTensor.shape();
        var contracted = Tensor.horizontalVector(1.0 / shape[0])
                             .broadcasted(1, shape[0])
                             .contracted(deltasTensor);
        return new Biases(
            tensor.difference(
                contracted.product(
                    Tensor.horizontalVector(learningRate.value())
                        .broadcasted(contracted.shape())
                )
            )
        );
    }

    int size() {
        var shape = tensor.shape();
        return shape[shape.length - 1];
    }
}
