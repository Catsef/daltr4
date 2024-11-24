
using System.Collections;
using System.Security.Cryptography;
using daltr4;

Console.WriteLine("hello bro - Written by Caltr4 - https://github.com/Catsef/daltr4");

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

public class Layer
{
    private bool IsHiddenLayer;
    private bool IsFinalLayer;
    private bool IsOutputLayer;
    private bool IsInputLayer;
    private int InputSize;
    private int OutputSize;
    private int Size;
    private double LearningRate;
    private Activation Activation;
    private double[][] weights;
    private double[] biases;
    private Nullable<double>[] inputs;
    
    public void Initialize()
    {
        if (IsHiddenLayer || IsFinalLayer)
        {
            weights = new double[InputSize][]; // inputsize, size; controls how each of the inputs affect the gen value
            for (var x = 0; x < InputSize; x++) {
                weights[x] = new double[Size];
                for (var i = 0; i < Size; i++)
                {
                    weights[x][i] = RandomNumberGenerator.GetInt32(0, 1);
                }
            }
            for (var i = 0; i < Size; i++)
            {
                if (!IsOutputLayer)
                {
                    biases[i] = RandomNumberGenerator.GetInt32(0, 1);
                }
                else
                {
                    break;
                }
            }
        } else if (IsInputLayer)
        {
            if (inputs == null)
            {
                throw new ArgumentException("Since you're declaring an input layer, you must provide inputs.");
            }
            InputSize = inputs.Length;
        }
    }

    public double[] ForwardPass(double[] inputs)
    {
        if (!IsInputLayer)
        {
            double[] neurons = new double[Size];
            for (var neuron = 0; neuron < neurons.Length; neuron++) { // running through each neuron in our neurons
                double Lneuron = biases[neuron]; // instead of adding the biases, we can directly set it as a default value
                if (IsOutputLayer)
                {
                    Lneuron = 0;
                }
                for (var input = 0; input < inputs.Length; input++) { // running through each input for the inputs of our selected neuron
                    Lneuron += inputs[input] * weights[input][neuron];
                }

                neurons[neuron] = Activation.activate(Lneuron);
            }

            return neurons;
        }
        else
        {
            return inputs;
        }
    }
    
    public (double[][], double[], double[], double[]) BackwardPass(
        double[] expectedOutputs,
        double[] inputs,
        double[] nextLayerDeltas,
        Layer nextLayer)
    {
        
        if (IsInputLayer) // if its a input
        {
            return (null, null, null, null); // theres nothing to be tweaked. So just return everything as null
        }

        double[][] deltaWeights = Initialize2DArray(InputSize, Size); // this will be what stores the weights that are to be changed and tweaked
        double[] deltaBiases = biases == null ? null : new double[Size]; // biases to be changed by value to improve the layer. can be null as it might be output
        double[] deltaNeurons = new double[Size]; // neurons that value's supposed to be changed by value
        double[] deltaInputs = IsOutputLayer ? null : new double[InputSize]; // inputs that are to be changed? by value

        double[] neurons = ForwardPass(inputs); // get the current layer's neuron output

        // Compute deltas
        if (IsOutputLayer) // if its an output layer
        {
            for (int i = 0; i < Size; i++) // loop through every delta neuron
            {
                deltaNeurons[i] = (expectedOutputs[i] - neurons[i]) * Activation.derivactivate(neurons[i]); // change it to equal to the distance between expected and real
            }
        }
        else // if not
        {
            for (int i = 0; i < Size; i++) // loop through inputs
            {
                double sum = 0;
                for (int j = 0; j < nextLayer.Size; j++) // loop through the neurons
                {
                    sum += nextLayerDeltas[j] * nextLayer.weights[i][j]; // add to the sum of the differences
                }
                deltaNeurons[i] = sum * Activation.derivactivate(neurons[i]); 
            }
        }

        // Update weights and biases
        for (int i = 0; i < InputSize; i++) 
        {
            for (int j = 0; j < Size; j++)
            {
                deltaWeights[i][j] = LearningRate * deltaNeurons[j] * inputs[i]; // to be applied when learn function
            }
        }

        if (deltaBiases != null)
        {
            for (int j = 0; j < Size; j++)
            {
                deltaBiases[j] = LearningRate * deltaNeurons[j]; // to be applied when learn function
            }
        }

        // Compute deltaInputs for hidden layers
        if (deltaInputs != null)
        {
            for (int i = 0; i < InputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < Size; j++)
                {
                    sum += deltaNeurons[j] * weights[i][j]; // negative gradient for neuron I in current layerL the sum of everything in J {negative gradient for neuron J in previous layer times the weight connecting neuron i to neuron J.
                }
                deltaInputs[i] = sum;
            }
        }

        return (deltaWeights, deltaBiases, deltaInputs, deltaNeurons);
    }

    private double[][] Initialize2DArray(int rows, int cols)
    {
        double[][] array = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            array[i] = new double[cols];
        }
        return array;
    }


    public double[][] getLtoCWeights()
    {
        return weights;
    }
    
    Layer (bool isHiddenLayer, bool isFinalLayer, bool isOutputLayer, 
        bool isInputLayer, int inputSize, int outputSize, 
        double learningRate, activationFunction activation, int layerSize, 
        int layerBias, Nullable<double>[] inputs)
    {
        IsHiddenLayer = isHiddenLayer;
        IsFinalLayer = isFinalLayer;
        IsOutputLayer = isOutputLayer;
        IsInputLayer = isInputLayer;
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = new Activation(activation);
        Size = layerSize;
        LearningRate = learningRate;
    }
}

