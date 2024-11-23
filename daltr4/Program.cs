
using System.Collections;
using System.Security.Cryptography;
using daltr4;

Console.WriteLine("hello bro - Made by Caltr4 - https://github.com/Catsef/daltr4");

public class Activation
{
    private activationFunction ActivationFunction;
    
    
    
    public Activation(activationFunction func)
    {
        ActivationFunction = func;
    }
    
    double ReLU (double x)
    {
        return Math.Max(0, x);
    }
    double Tanh(double x)
    {
        return 2/(1+Math.Pow(Math.E, -(2*x)))-1;
    }
    private double BipolarSigmoid(double x)
    {
        return (1-Math.Exp(-x))/(1+Math.Exp(-x));
    }

    private double UnReLU(double x)
    {
        if (x < 0) return 0;
        else return 1;
    }

    private double UnTanh(double x)
    {
        double e = Math.E;
        return 1 - Math.Pow(Math.Exp(x) - Math.Exp(-x), 2) / Math.Pow(Math.Exp(x) + Math.Exp(-x), 2);
    }

    private double UnBipolarSigmoid(double x)
    {
        return 1 - Math.Pow(Tanh(x / 2), 2);
    }
    public double activate(double x)
    {
        if (ActivationFunction == activationFunction.ReLu) return ReLU(x);
        if (ActivationFunction == activationFunction.Tanh) return Tanh(x);
        if (ActivationFunction == activationFunction.Sigmoid) return BipolarSigmoid(x);
        else return 0;
    }

    public double derivactivate(double x)
    {
        if (ActivationFunction == activationFunction.ReLu) return UnReLU(x);
        if (ActivationFunction == activationFunction.Tanh) return UnTanh(x);
        if (ActivationFunction == activationFunction.Sigmoid) return UnBipolarSigmoid(x);
        else return 0;
    }
}

public class DenseHiddenLayer
{
    private int InputSize; // previous layer's size
    private int OutputSize; // next layer's size
    private int Size;
    private double LearningRate; // tWEAKUNG SIZE
    private Activation Activation;
    private double[][] weights; // weights [Input] [Neuron]
    private double[] biases;

    public void Initialize()
    {
        
        weights = new double[InputSize][]; // inputsize, size; controls how each of the inputs affect the gen value
        for (var x = 0; x < InputSize; x++) {
            weights[x] = new double[Size];
            for (var i = 0; i < Size; i++)
            {
                weights[x][i] = RandomNumberGenerator.GetInt32(0, 1);
            }
        }
        biases = new double[Size]; // its the intercept - offsets values. Neuron V = (Previous Layer : Sum : ReLU * weights) + bias.

        for (var i = 0; i < Size; i++)
        {
            biases[i] = RandomNumberGenerator.GetInt32(0, 1);
        }
    }
    
    public double[] ForwardPass(double[] inputs)
    {
        double[] neurons = new double[Size];
        for (var neuron = 0; neuron < neurons.Length; neuron++) { // running through each neuron in our neurons
            double Lneuron = biases[neuron]; // instead of adding the biases, we can directly set it as a default value
            for (var input = 0; input < inputs.Length; input++) { // running through each input for the inputs of our selected neuron
                Lneuron += inputs[input] * weights[input][neuron];
            }

            neurons[neuron] = Activation.activate(Lneuron);
        }

        return neurons;
    }

    public (double[][], double[], double[], double[]) 
        BackwardPass(
            double[] OutputExpectedValues, 
            double[] inputs,
            double[] nextLayerDeltas, 
            DenseHiddenLayer nextLayer, 
            DenseHiddenLayer prevLayer
            )
    {
        double[][] DeltaWeights = new double[InputSize][];
        for (int i = 0; i < InputSize; i++)
        {
            DeltaWeights[i] = new double[Size];
        }
        double[] DeltaBiases = new double[Size];
        double[] DeltaInputs = new double[InputSize];

        double[] Neurons = new double[Size];
        Neurons = ForwardPass(inputs);
        double[] DeltaNeurons = new double[Size];
        for (int neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
        {
            double SUM = 0;
            for (int NextLayerDeltaIndex = 0; NextLayerDeltaIndex < nextLayerDeltas.Length; NextLayerDeltaIndex++)
            {
                SUM += nextLayerDeltas[NextLayerDeltaIndex] * nextLayer.getLtoCWeights()[neuronIndex][NextLayerDeltaIndex];
            }
            
            DeltaNeurons[neuronIndex] = Activation.derivactivate(Neurons[neuronIndex]) * SUM;
        }
        
        for (var neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
        {
            for (var input = 0; input < inputs.Length; input++)
            {
                DeltaWeights[input][neuronIndex] = LearningRate * DeltaNeurons[neuronIndex] * inputs[input];
            }

            DeltaBiases[neuronIndex] = LearningRate * DeltaNeurons[neuronIndex];
        }
        return (DeltaWeights, DeltaBiases, DeltaInputs, DeltaNeurons);
    }

    public double[][] getLtoCWeights()
    {
        return weights;
    }
    
    
    
    DenseHiddenLayer (int inputSize, int outputSize, int layerSize, double learningRate, activationFunction activationFunction)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = new Activation(activationFunction);
        Size = layerSize;
        LearningRate = learningRate;
        
    }
}