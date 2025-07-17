package com.github.maximtereshchenko.snapdragon;

import java.util.List;
import java.util.function.Supplier;

final class ValidationDataset extends Dataset {

    ValidationDataset(Supplier<List<LabeledSample>> supplier, int batchSize) {
        super(supplier, batchSize);
    }

    @Override
    List<LabeledSample> labeledSamples(Supplier<List<LabeledSample>> supplier) {
        return supplier.get();
    }
}
