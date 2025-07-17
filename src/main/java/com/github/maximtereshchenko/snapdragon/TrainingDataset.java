package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

final class TrainingDataset extends Dataset {

    private final Random random;

    TrainingDataset(
        Supplier<List<LabeledSample>> supplier,
        int batchSize,
        Random random
    ) {
        super(supplier, batchSize);
        this.random = random;
    }

    @Override
    List<LabeledSample> labeledSamples(Supplier<List<LabeledSample>> supplier) {
        var labeledSamples = new ArrayList<>(supplier.get());
        Collections.shuffle(labeledSamples, random);
        return labeledSamples;
    }
}
