package com.github.maximtereshchenko.snapdragon;

record Weights(Tensor tensor) {

    Weights calibrated(Outputs outputs, Deltas deltas, LearningRate learningRate) {
        var deltasTensor = deltas.tensor();
        var product = outputs.tensor().transposed().contracted(deltasTensor);
        return new Weights(
            tensor.difference(
                product.product(
                    Tensor.horizontalVector(
                            learningRate.value() / deltasTensor.shape().getFirst()
                        )
                        .broadcasted(product.shape()))
            )
        );
    }
}
