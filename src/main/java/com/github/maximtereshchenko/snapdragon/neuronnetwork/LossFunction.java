package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;

public interface LossFunction {

    Matrix derivative(Matrix outputs, Matrix labels);
}
