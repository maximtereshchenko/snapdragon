package com.github.maximtereshchenko.snapdragon;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
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

    List<double[]> data(Path path) throws IOException {
        try (var inputStream = Files.newInputStream(path)) {
            var data = new ArrayList<double[]>();
            while (inputStream.available() > 0) {
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
                var values = new double[length(inputStream, dimensions)];
                for (var i = 0; i < values.length; i++) {
                    values[i] = inputStream.read();
                }
                data.add(values);
            }
            return data;
        }
    }

    private List<LabeledSample> labeledSamples() {
        try {
            var inputs = data(inputsPath);
            var labels = data(labelsPath).getFirst();
            var labeledSamples = new ArrayList<LabeledSample>();
            for (var i = 0; i < inputs.size(); i++) {
                var oneHotEncodedLabels = new double[10];
                oneHotEncodedLabels[(int) labels[i]] = 1;
                labeledSamples.add(new StaticLabeledSample(inputs.get(i), oneHotEncodedLabels));
            }
            return labeledSamples;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private int length(InputStream inputStream, int dimensions) throws IOException {
        var length = 1;
        for (var i = 0; i < dimensions; i++) {
            var buffer = new byte[4];
            if (inputStream.read(buffer) != buffer.length) {
                throw new IllegalStateException();
            }
            length *= ByteBuffer.wrap(buffer).getInt();
        }
        return length;
    }
}
