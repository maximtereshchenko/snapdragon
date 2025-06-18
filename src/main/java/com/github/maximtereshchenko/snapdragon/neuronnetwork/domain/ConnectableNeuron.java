package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

interface ConnectableNeuron {

    Connection connection(int destinationIndex);
}
