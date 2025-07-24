package com.github.maximtereshchenko.snapdragon;

import java.util.Objects;

final class HiddenLayer extends ForwardPropagationParticipant<HiddenLayer> {

    HiddenLayer(LayerIndex index, Biases biases, ActivationFunction activationFunction) {
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
        return object instanceof HiddenLayer &&
                   super.equals(object);
    }

    @Override
    HiddenLayer calibrated(
        LayerIndex index,
        Biases calibrated,
        ActivationFunction activationFunction
    ) {
        return new HiddenLayer(index, calibrated, activationFunction);
    }

    Deltas deltas(Deltas deltas, Outputs outputs, Weights weights) {
        return new Deltas(
            activationFunction().deltas(
                outputs.tensor(),
                deltas.tensor().contracted(weights.tensor().transposed())
            )
        );
    }
}
