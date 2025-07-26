package com.github.maximtereshchenko.snapdragon;

import java.util.Arrays;

record StaticLabeledSample(double[] inputs, double[] labels) implements LabeledSample {

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof StaticLabeledSample that &&
                   Arrays.equals(inputs, that.inputs) &&
                   Arrays.equals(labels, that.labels);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(inputs);
        result = 31 * result + Arrays.hashCode(labels);
        return result;
    }

    @Override
    public String toString() {
        return "StaticLabeledSample{" +
                   "inputs=" + Arrays.toString(inputs) +
                   ", labels=" + Arrays.toString(labels) +
                   '}';
    }
}
