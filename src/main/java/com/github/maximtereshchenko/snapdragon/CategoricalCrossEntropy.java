package com.github.maximtereshchenko.snapdragon;

record CategoricalCrossEntropy() implements LossFunction {

    @Override
    public Tensor loss(Tensor outputs, Tensor labels) {
        var shape = outputs.shape();
        return labels.product(Tensor.from(shape, index -> Math.log(outputs.value(index))))
                   .contracted(
                       Tensor.verticalVector(-1).broadcasted(shape.getLast(), 1)
                   );
    }

    @Override
    public Tensor derivative(Tensor outputs, Tensor labels) {
        var shape = labels.shape();
        return labels.product(Tensor.horizontalVector(-1).broadcasted(shape))
                   .quotient(outputs);
    }
}
