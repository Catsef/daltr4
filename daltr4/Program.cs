
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
    
    private int InputSize; // previous layer's size or CNN input size
    private int OutputSize;
    private int Size;
    private double LearningRate;
    private activationFunction ActivationFunction;
    private double[][] weights; // weights [Node] [Input]
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
        for (var neuron = 0; neuron < Size; neuron++) { // running through each neuron in our neurons
            var Lneuron = neurons[neuron];
            for (var input = 0; input < inputs.Length; input++) { // running through each input for the inputs of our selected neuron
                Lneuron += inputs[input] * weights[neuron][input];
            }

            neurons[neuron] = Lneuron;
        }

        return neurons;
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