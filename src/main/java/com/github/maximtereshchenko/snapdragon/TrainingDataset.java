package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

final class TrainingDataset extends Dataset {

    TrainingDataset(List<? extends LabeledSample> original) {
        super(original);
    }

    @Override
    List<LabeledSample> labeledSamples(List<? extends LabeledSample> original) {
        var copy = new ArrayList<LabeledSample>(original);
        Collections.shuffle(copy);
        return copy;
    }
}
