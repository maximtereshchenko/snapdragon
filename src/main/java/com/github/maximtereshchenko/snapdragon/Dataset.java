package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.function.Supplier;

abstract class Dataset {

    private final Supplier<List<LabeledSample>> supplier;
    private final int batchSize;

    Dataset(Supplier<List<LabeledSample>> supplier, int batchSize) {
        this.supplier = supplier;
        this.batchSize = batchSize;
    }

    final Iterator<BatchedLabeledSample> batchedLabeledSamples() {
        return new BatchedLabeledSampleIterator(labeledSamples(supplier).iterator(), batchSize);
    }

    abstract List<LabeledSample> labeledSamples(
        Supplier<List<LabeledSample>> supplier
    );

    private static final class BatchedLabeledSampleIterator
        implements Iterator<BatchedLabeledSample> {

        private final Iterator<LabeledSample> labeledSamples;
        private final int batchSize;
        private BatchedLabeledSample batchedLabeledSample;

        BatchedLabeledSampleIterator(
            Iterator<LabeledSample> labeledSamples,
            int batchSize
        ) {
            this.labeledSamples = labeledSamples;
            this.batchSize = batchSize;
        }

        @Override
        public boolean hasNext() {
            if (batchedLabeledSample == null) {
                tryAdvance();
            }
            return batchedLabeledSample != null;
        }

        @Override
        public BatchedLabeledSample next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            var next = batchedLabeledSample;
            batchedLabeledSample = null;
            return next;
        }

        private void tryAdvance() {
            var samples = new ArrayList<LabeledSample>();
            while (labeledSamples.hasNext() && samples.size() < batchSize) {
                samples.add(labeledSamples.next());
            }
            if (samples.isEmpty()) {
                return;
            }
            var inputs = new double[samples.size()][];
            var labels = new double[samples.size()][];
            for (var i = 0; i < samples.size(); i++) {
                var current = samples.get(i);
                inputs[i] = current.inputs();
                labels[i] = current.labels();
            }
            batchedLabeledSample = new BatchedLabeledSample(
                new Inputs(tensor(inputs)),
                new Labels(tensor(labels))
            );
        }

        private Tensor tensor(double[][] array) {
            return Tensor.from(
                new int[]{array.length, array[0].length},
                index -> array[index[0]][index[index.length - 1]]
            );
        }
    }
}
