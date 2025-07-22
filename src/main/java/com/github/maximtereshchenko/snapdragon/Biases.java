package com.github.maximtereshchenko.snapdragon;

record Biases(Tensor tensor) {

    Biases calibrated(Deltas deltas, LearningRate learningRate) {
        var deltasTensor = deltas.tensor();
        var shape = deltasTensor.shape();
        var contracted = Tensor.horizontalVector(1.0 / shape.getFirst())
                             .broadcasted(1, shape.getFirst())
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
        return tensor.shape().getLast();
    }
}
