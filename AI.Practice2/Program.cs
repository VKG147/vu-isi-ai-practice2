using AI.Practice1;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AI.Practice2
{
    public static class Program
    {
        private const decimal DATA_SPLIT_RATIO = 0.8m;
        private const int RAND_SEED = 111;

        private static readonly Random random = new(RAND_SEED);

        public static void Main()
        {
            //List<float[]> data = GetData("Data/iris_2.data");
            //DataProcessor dataProcessor = new(random);
            //(List<float[]> trainingData, List<float[]> testingData) = dataProcessor.SplitData(data, DATA_SPLIT_RATIO);
            //Neuron neuron = new SigmoidNeuron(trainingData[0].Length);
            //NeuronSupervisor.InitializeNeuronWeights(neuron, random);
            //float learningRate = 2f;
            //int iterations = 200;

            // ===0.PREPARATION===
            // Read data
            List<float[]> data = GetData();

            // Split data into training and testing sets
            DataProcessor dataProcessor = new(random);
            (List<float[]> trainingData, List<float[]> testingData) = dataProcessor.SplitData(data, DATA_SPLIT_RATIO);

            // Get neuron with user chosen activation function
            Neuron neuron = GetNeuron(trainingData[0].Length);
            NeuronSupervisor.InitializeNeuronWeights(neuron, random);

            // Learning rate and iteration number
            (float learningRate, int iterations) = GetTrainingParameters();

            trainingData = DataProcessor.PrependBiasInputs(trainingData);
            testingData = DataProcessor.PrependBiasInputs(testingData);

            // ===1.LEARNING===
            for (int i = 0, j = 0; i < iterations; ++i)
            {
                if (j >= trainingData.Count) j = 0;
                float[] inputs = trainingData[j].SkipLast(1).ToArray();
                NeuronSupervisor.UpdateWeights(neuron, inputs, trainingData[j].Last(), learningRate);
                j++;
            }

            // ===2.TESTING===
            int correctPredictions = 0;
            float totalE = 0f;
            float minE = float.MaxValue;
            float maxE = float.MinValue;

            foreach (float[] values in testingData)
            {
                float calculated = neuron.Compute(values.SkipLast(1).ToArray());
                float expected = values.Last();

                float e = MathF.Pow(calculated - expected, 2);
                totalE += e;
                minE = MathF.Min(minE, e);
                maxE = MathF.Max(maxE, e);

                if (MathF.Round(calculated) == expected) correctPredictions++;
            }
            
            // ===3.RESULTS===
            Console.Clear();
            Console.WriteLine("===PARAMETERS===");
            Console.WriteLine($"Neuron type: {neuron.GetType().Name}");
            Console.WriteLine($"Learning rate: {learningRate}");
            Console.WriteLine($"Number of iterations: {iterations}");

            Console.WriteLine("\n===WEIGHTS===");
            for (int i = 0; i < neuron.Weights.Length; ++i)
            {
                Console.WriteLine($"w{i} = {neuron.Weights[i]}");
            }
            Console.WriteLine("\n===ACCURACY===");
            Console.WriteLine($"Prediction ratio: {correctPredictions}/{testingData.Count} ({100.0f * correctPredictions / testingData.Count}%)");
            Console.WriteLine("\n===COST FUNCTION===");
            Console.WriteLine($"Average cost: {totalE / testingData.Count}");
            Console.WriteLine($"Min cost: {minE}");
            Console.WriteLine($"Max cost: {maxE}");
        }

        public static List<float[]> GetData(string path = null)
        {
            while (!File.Exists(path))
            {
                Console.Clear();
                Console.WriteLine("Please enter data file path:");
                path = Console.ReadLine();
            }
            
            using DataReader reader = new(path);
            return reader.ReadAll();
        }

        public static Neuron GetNeuron(int weightCount)
        {
            string neuronType;
            while (true)
            {
                Console.Clear();
                Console.WriteLine("Select activation function type:");
                Console.WriteLine("a - sigmoid");
                Console.WriteLine("b - binary stop");
                neuronType = Console.ReadLine();

                if (neuronType.ToLower() == "a") return new SigmoidNeuron(weightCount);
                else if (neuronType.ToLower() == "b") return new BinaryStopNeuron(weightCount);
            }
        }

        public static (float learningRate, int iterations) GetTrainingParameters()
        {
            float? learningRate = null;
            do
            {
                Console.Clear();
                Console.WriteLine("Enter learning rate (floating point number):");
                string input = Console.ReadLine();
                if (float.TryParse(input, out float result)) learningRate = result;
            }
            while (learningRate is null);

            int? iterations = null;
            do
            {
                Console.Clear();
                Console.WriteLine("Enter number of iterations:");
                string input = Console.ReadLine();
                if (int.TryParse(input, out int result)) iterations = result;
            }
            while (iterations is null || iterations < 0);

            return new(learningRate ?? default, iterations ?? default);
        }
    }
}
