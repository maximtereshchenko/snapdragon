package com.github.maximtereshchenko.snapdragon;

import java.util.List;

final class ControlDataset extends Dataset {

    ControlDataset(List<? extends LabeledSample> original) {
        super(original);
    }

    @Override
    List<LabeledSample> labeledSamples(List<? extends LabeledSample> original) {
        return List.copyOf(original);
    }
}
