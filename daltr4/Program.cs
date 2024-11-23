
using System.Security.Cryptography;
using daltr4;

Console.WriteLine("hello bro - Made by Caltr4 - https://github.com/Catsef/daltr4");



public class DenseLayer
{
    
    double ReLU (double x)
    {
        return Math.Max(0, x);
    }
    double Tanh(double x)
    {
        return 2/(1+Math.Pow(Math.E, -(2*x)))-1;
    }
    double BipolarSigmoid(double x)
    {
        return (1-Math.Exp(-x))/(1+Math.Exp(-x));
    }
    double activate(double x)
    {
        if (ActivationFunction == activationFunction.ReLu) return ReLU(x);
        if (ActivationFunction == activationFunction.Tanh) return Tanh(x);
        if (ActivationFunction == activationFunction.Sigmoid) return BipolarSigmoid(x);
        else return 0;
    }
    
    private int InputSize; // previous layer's size or CNN input size
    private int OutputSize;
    private int Size;
    private double LearningRate; // tWEAKUNG SIZE
    private activationFunction ActivationFunction;
    private double[][] weights; // weights [Input] [Neuron]
    private double[] biases;

    public void Initialize()
    {
        
        weights = new double[InputSize][]; // inputsize, size; controls how each of the inputs affect the gen value
        for (var x = 0; x < InputSize; x++) {
            weights[x] = new double[Size];
            for (var i = 0; i < Size; i++)
            {
                weights[x][i] = RandomNumberGenerator.GetInt32(-1, 1);
            }
        }
        biases = new double[Size]; // its the intercept - offsets values. Neuron V = (Previous Layer : Sum : ReLU * weights) + bias.

        for (var i = 0; i < Size; i++)
        {
            biases[i] = RandomNumberGenerator.GetInt32(-1, 1);
        }
    }
    
    public double[] ForwardPass(double[] inputs)
    {
        double[] neurons = new double[Size];
        for (var neuron = 0; neuron < neurons.Length; neuron++) { // running through each neuron in our neurons
            double Lneuron = biases[neuron]; // instead of adding the biases, we can directly set it as a default value
            for (var input = 0; input < inputs.Length; input++) { // running through each input for the inputs of our selected neuron
                Lneuron += inputs[input] * weights[neuron][input];
            }

            neurons[neuron] = activate(Lneuron);
        }

        return neurons;
    }

    public void BackwardPass(double[] ExpectedValues, double[] inputs)
    {
        double[][] DeltaWeights = new double[InputSize][];
        double[] DeltaBiases = new double[Size];
        double[] DeltaInputs = new double[InputSize];

        double[] Neurons = new double[Size];
        Neurons = ForwardPass(inputs);
        
        double[] errors = new double[Size];
        for (var neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
        {
            errors[neuronIndex] = ExpectedValues[neuronIndex] - Neurons[neuronIndex];
            // delta Neuron = Error * derivative of activation function
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