

// Backpropagation algorithm:
// Inputs all arrive through the preinitialised paths
// weights are chosen randomly
// Calculate the output of each neuron from the input layer
// to the hidden layer to the output layer
// Calculate the error in the output layer
// From the output layer, go back to the hidden layer to adjust the weights
// so that it reduces the error
// Repeat the process until the desired output is achieved.

// Initialise weights to small random values
// While we're not stopping the stepping, do the following
// For each training pair (data) do the following
// Calculate and populate the output by sending and activating
// Each output unit "OutputUnitK"
// (K is index of unit from 1 to the amount of outputs N)
// recieves a target output corresponding to an given input
// and the error between the real output and the target output
// is calculated as 
// Error K = (Target Output K - Real Output K)
// Each neuron in the hidden layer (TotalLayers - 1) as J sums
// basic example
// In a classifier model, we have 5 neurons in the output layer.
// we will call this layer L. the softmax value of each output
// neuron represents the likelihood out of 1 that the input
// belongs to their category.
// For now, we'll say that the output unit is representing the perfect
// and correct prediction, which we'll call L<c>.
// L<c>'s activation function is a composite function, containing
// the many nested activation functions of the entire neural network
// from the input to the output.
// Minimizing the loss would entail making adjustments throughout the
// network that bring the output of L<c>'s error closer to 0.
// to do so, we'll need to know how any change in previous layers will
// change L<c>'s output, which means that
// we will have to find the partial derivatives of L<c>'s activation
// function.

using daltr4;
using MathNet.Numerics;

public class ActivationUtility
{
    private activationFunction ActivationFunction;
    
    public ActivationUtility(activationFunction func)
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
class DenseLayer
{
    public int amountOfNeurons;
    
    public bool isInputLayer;
    public bool isOutputLayer;
    public bool isDenseLayer;
    public double[][]? weights; // -o. connections from the layer behind. [BEHIND], [CURRENT]
    public double[]? biases; // +bias
    public double[]? inputs;
    public double? learningRate;
    public double[]? input;
    public ActivationUtility activation;

    public bool UpdateInput(double[] input)
    {
        if (!isInputLayer) return false;
        this.input = input;
        return true;
    }
    
    // TODO: Initialize function?
    // TODO: Update Bias function?
    
    // TODO: Remove all manual for loops and replace with MathNET Numerics Matrix
    
    public double[] ForwardPass(double[]? lastLayerNeurons)
    {
        if ((isDenseLayer || isOutputLayer) && lastLayerNeurons == null) 
        {
            throw new ArgumentException("Must pass last layer neurons to forward pass!");
        } else if (isInputLayer)
        {
            return input;
        }
        double[] Neurons = new double[amountOfNeurons];
        for (var NeuronIndex = 0; NeuronIndex < Neurons.Length; NeuronIndex++)
        {
            var Neuron = biases[NeuronIndex]; // add the bias
            if (isOutputLayer)
            {
                // if its the output layer, no bias should be added
                Neuron = 0;
            }

            for (var LastLayerNeuronIndex = 0;
                 LastLayerNeuronIndex < lastLayerNeurons.Length;
                 LastLayerNeuronIndex++)
            {
                // the sum of all previous layer neurons times the weight
                Neuron += lastLayerNeurons[LastLayerNeuronIndex] * weights[LastLayerNeuronIndex][NeuronIndex]; // j-i
            }
            // put through the activation function
            Neuron = activation.activate(Neuron);
            Neurons[NeuronIndex] = Neuron;

        }
        return Neurons;
    }

    public (double[][], double[]) BackwardPass(double[] ExpectedOutputOrNextErrorTerms, double[] Input, DenseLayer? NextLayer, DenseLayer? PreviousLayer)
    {
        if (isOutputLayer)
        {
            if (PreviousLayer == null)
            {
                throw new ArgumentException("Must pass previous layer to backward pass if output!");
                return (null, null);
            }
            double[] GottenOutputs = new double[amountOfNeurons];
            GottenOutputs = ForwardPass(Input);
            double[] ErrorTermsForOutput = new double[amountOfNeurons];
            if (ExpectedOutputOrNextErrorTerms.Length != ErrorTermsForOutput.Length)
            {
                throw new Exception("Expected output length needs to fit to output node length!");
                return (null, null);
            }
            for (var OutputIndex = 0; OutputIndex < ErrorTermsForOutput.Length; OutputIndex++)
            {
                ErrorTermsForOutput[OutputIndex] = (ExpectedOutputOrNextErrorTerms[OutputIndex] - GottenOutputs[OutputIndex])
                                                   * activation.derivactivate(GottenOutputs[OutputIndex]);
            }

            return (null, ErrorTermsForOutput);
        } else if (isDenseLayer)
        {
            if (NextLayer == null)
            {
                throw new ArgumentException("Must pass next layer to backward pass!");
                return (null, null);
            }
            double[] GottenOutputs = new double[amountOfNeurons];
            GottenOutputs = ForwardPass(Input);
            
            double[] ErrorTermsForCurrentLayer = new double[amountOfNeurons];
            for (var CurrentNeuronIndex = 0; CurrentNeuronIndex < amountOfNeurons; CurrentNeuronIndex++)
            {
                double SumOfAllErrorsOfNextLayer = 0;
                for (var CurrentNextErrorIndex = 0; CurrentNextErrorIndex < ExpectedOutputOrNextErrorTerms.Length; CurrentNextErrorIndex++)
                {
                    SumOfAllErrorsOfNextLayer += ExpectedOutputOrNextErrorTerms[CurrentNextErrorIndex] *
                                                 NextLayer.weights[CurrentNeuronIndex][CurrentNextErrorIndex];
                }
                SumOfAllErrorsOfNextLayer = activation.derivactivate(SumOfAllErrorsOfNextLayer);
                ErrorTermsForCurrentLayer[CurrentNeuronIndex] = SumOfAllErrorsOfNextLayer;
            }
            
            double[][] WeightChanges = new double[Input.Length][];
            for (var i=0;i<Input.Length;i++)
            {
                WeightChanges[i] = new double[amountOfNeurons];
            }
            
            for (var neuron = 0; neuron < amountOfNeurons; neuron++)
            {
                for (var input = 0; input < Input.Length; input++)
                {
                    WeightChanges[input][neuron] = (double)(-learningRate * ErrorTermsForCurrentLayer[neuron] * Input[input]);
                }
                biases[neuron] = (double) (-learningRate * ErrorTermsForCurrentLayer[neuron]);
            }
            
            return (WeightChanges, ErrorTermsForCurrentLayer);
        } else if (isInputLayer)
        {
            return (null, null);
        }

        return (null, null);
    }

    public void updateWeights(double[][] weights, double[] Grad)
    {
        // TODO: Finish
    }

    public DenseLayer(bool input, bool output, bool dense, int neurons, activationFunction function, double? learningRate, double[]? inputs)
    {
        if (isDenseLayer && learningRate == null)
        {
            learningRate = 0;
            Console.WriteLine("Dense layer needs Learning rate to be set! returning to 0.");
            
        }

        if (!isInputLayer && inputs != null)
        {
            throw new ArgumentException("Dense or output layers don't need inputs!");
        } else if (isInputLayer && inputs == null)
        {
            throw new ArgumentException("Input layers need inputs!");
        }
        else
        {
            this.input = inputs;
        } 
        activation = new ActivationUtility(function);
        isInputLayer = input;
        isOutputLayer = output;
        isDenseLayer = dense;
        amountOfNeurons = neurons;
        
    }
}

class CNN
{
    public DenseLayer[] layers;
    public DenseLayer InputLayer;
    public DenseLayer OutputLayer;

    public CNN(DenseLayer[] layers, double learningRate, double[] defaultInputs)
    {
        if (!layers[0].isInputLayer) throw new ArgumentException("1st layer must be input layer!");
        if (!layers[-1].isOutputLayer) throw new ArgumentException("Last layer must be output layer!");
        
        
    }
}