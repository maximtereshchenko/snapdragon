package com.github.maximtereshchenko.snapdragon;

import java.util.List;

final class ValidationDataset extends Dataset {

    ValidationDataset(List<? extends LabeledSample> original) {
        super(original);
    }

    @Override
    List<LabeledSample> labeledSamples(List<? extends LabeledSample> original) {
        return List.copyOf(original);
    }
}
