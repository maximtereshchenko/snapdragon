package com.github.maximtereshchenko.snapdragon;

import java.util.Objects;

final class OutputLayer extends ForwardPropagationParticipant<OutputLayer> {

    OutputLayer(LayerIndex index, Biases biases, ActivationFunction activationFunction) {
        super(index, biases, activationFunction);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode());
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof OutputLayer &&
                   super.equals(object);
    }

    @Override
    OutputLayer calibrated(
        LayerIndex index,
        Biases calibrated,
        ActivationFunction activationFunction
    ) {
        return new OutputLayer(index, calibrated, activationFunction);
    }

    Deltas deltas(LayerMap<Outputs> outputs, LossFunction lossFunction, Labels labels) {
        var outputsTensor = outputs.element(index()).tensor();
        return new Deltas(
            activationFunction().deltas(
                outputsTensor,
                lossFunction.derivative(outputsTensor, labels.tensor())
            )
        );
    }
}
