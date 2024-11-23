﻿// See https://aka.ms/new-console-template for more information

using System.Security.Cryptography;
using daltr4;

Console.WriteLine("hello bro - Made by Caltr4 - https://github.com/Catsef/daltr4");
double Tanh(double x)
{
    return 2/(1+Math.Pow(Math.E, -(2*x)))-1;
}

double BipolarSigmoid(double x)
{
    return (1-Math.Exp(-x))/(1+Math.Exp(-x));
}

public class DenseLayer
{
    
    double ReLU (double x)
    {
        return Math.Max(0, x);
    }
    
    private int InputSize;
    private int OutputSize;
    private int Size;
    private double LearningRate;
    private activationFunction ActivationFunction;
    private double[] neurons;
    private double[][] weights; // weights [Node] [Input]
    private double[] biases;

    public void Initialize()
    {
        neurons = new double[Size]; // neurons - useless?
        
        weights = new double[InputSize][]; // inputsize, size; controls how each of the inputs affect the gen value
        for (var x = 0; x < InputSize; x++) {
            weights[x] = new double[Size];
            for (var i = 0; i < Size; i++)
            {
                weights[x][i] = RandomNumberGenerator.GetInt32(-1, 1);
            }
        }
        biases = new double[Size]; // basically C in Y=mX+C - its the intercept.

        for (var i = 0; i < Size; i++)
        {
            neurons[i] = RandomNumberGenerator.GetInt32(-1, 1);
            biases[i] = RandomNumberGenerator.GetInt32(-1, 1);
            
        }
    }
    
    public void ForwardPass(double[] inputs)
    {
        var index = 0;
        foreach (var neuron in neurons) {
            var Lneuron = neuron;
            foreach (var input in inputs) {
                Lneuron += input;
            }

            neurons[index] = Lneuron;
            index++;
        }
    }
    
    DenseLayer (int inputSize, int outputSize, int layerSize, double learningRate, activationFunction activationFunction)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        ActivationFunction = activationFunction;
        Size = layerSize;
        LearningRate = learningRate;
        
    }
}