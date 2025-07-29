package com.github.maximtereshchenko.snapdragon;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

final class MnistSamples<T extends Dataset> {

    private final Path inputsPath;
    private final Path labelsPath;
    private final Function<Supplier<List<LabeledSample>>, T> function;

    MnistSamples(
        Path inputsPath,
        Path labelsPath,
        Function<Supplier<List<LabeledSample>>, T> function
    ) {
        this.inputsPath = inputsPath;
        this.labelsPath = labelsPath;
        this.function = function;
    }

    T dataset() {
        return function.apply(this::labeledSamples);
    }

    <R> List<R> data(Path path, BiFunction<InputStream, Integer, R> function) throws IOException {
        try (var inputStream = Files.newInputStream(path)) {
            var data = new ArrayList<R>();
            var firstMagicNumber = inputStream.read();
            var secondMagicNumber = inputStream.read();
            var datatype = inputStream.read();
            var dimensions = inputStream.read();
            if (firstMagicNumber != 0 &&
                    secondMagicNumber != 0 &&
                    datatype != 8 && //unsigned byte
                    dimensions < 1) {
                throw new IllegalStateException();
            }
            var size = readInteger(inputStream);
            var length = length(inputStream, dimensions - 1);
            for (var currentItem = 0; currentItem < size; currentItem++) {
                data.add(function.apply(inputStream, length));
            }
            return data;
        }
    }

    private List<LabeledSample> labeledSamples() {
        try {
            var inputs = data(inputsPath, this::inputs);
            var labels = data(labelsPath, this::oneHotEncodedLabels);
            var labeledSamples = new ArrayList<LabeledSample>();
            for (var i = 0; i < inputs.size(); i++) {
                labeledSamples.add(new StaticLabeledSample(inputs.get(i), labels.get(i)));
            }
            return labeledSamples;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private double[] inputs(InputStream inputStream, int length) {
        try {
            var values = new double[length];
            for (var currentValue = 0; currentValue < values.length; currentValue++) {
                values[currentValue] = inputStream.read() / 255.0;
            }
            return values;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private double[] oneHotEncodedLabels(InputStream inputStream, int length) {
        try {
            if (length != 1) {
                throw new IllegalStateException();
            }
            var oneHotEncodedLabels = new double[10];
            oneHotEncodedLabels[inputStream.read()] = 1;
            return oneHotEncodedLabels;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private int length(InputStream inputStream, int dimensions) throws IOException {
        var length = 1;
        for (var i = 0; i < dimensions; i++) {
            length *= readInteger(inputStream);
        }
        return length;
    }

    private int readInteger(InputStream inputStream) throws IOException {
        var buffer = new byte[4];
        if (inputStream.read(buffer) != buffer.length) {
            throw new IllegalStateException();
        }
        return ByteBuffer.wrap(buffer).getInt();
    }
}
