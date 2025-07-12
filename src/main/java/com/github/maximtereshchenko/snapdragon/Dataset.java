package com.github.maximtereshchenko.snapdragon;

import java.util.List;

abstract class Dataset {

    private final List<? extends LabeledSample> original;

    Dataset(List<? extends LabeledSample> original) {
        this.original = original;
    }

    final LabeledSamples labeledSamples() {
        var labeledSamples = labeledSamples(original);
        var inputs = new double[labeledSamples.size()][];
        var labels = new double[labeledSamples.size()][];
        for (var i = 0; i < labeledSamples.size(); i++) {
            var labeledSample = labeledSamples.get(i);
            inputs[i] = labeledSample.inputs();
            labels[i] = labeledSample.labels();
        }
        return new LabeledSamples(new Inputs(Matrix.from(inputs)), new Labels(Matrix.from(labels)));
    }

    abstract List<LabeledSample> labeledSamples(List<? extends LabeledSample> original);
}
