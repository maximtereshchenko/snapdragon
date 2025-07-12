package com.github.maximtereshchenko.snapdragon;

record Weights(Matrix matrix) {

    Weights calibrated(Outputs outputs, Deltas deltas, LearningRate learningRate) {
        var deltasMatrix = deltas.matrix();
        return new Weights(
            matrix.combined(
                outputs.matrix()
                    .transposed()
                    .product(deltasMatrix)
                    .applied(value -> value / deltasMatrix.rows())
                    .applied(value -> learningRate.value() * value),
                (a, b) -> a - b
            )
        );
    }
}
