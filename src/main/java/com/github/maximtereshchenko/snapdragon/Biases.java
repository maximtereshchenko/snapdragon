package com.github.maximtereshchenko.snapdragon;

record Biases(Matrix matrix) {

    Biases calibrated(Deltas deltas, LearningRate learningRate) {
        var deltasMatrix = deltas.matrix();
        return new Biases(
            matrix.combined(
                Matrix.horizontalVector(1.0 / deltasMatrix.rows())
                    .broadcasted(1, deltasMatrix.rows())
                    .product(deltasMatrix)
                    .applied(sum -> learningRate.value() * sum),
                (a, b) -> a - b
            )
        );
    }

    int size() {
        return matrix.columns();
    }
}
