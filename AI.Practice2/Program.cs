using AI.Practice1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI.Practice2
{
    public static class Program
    {
        public const string DATA_PATH_ONE = "./Data/iris_1.data";
        public const string DATA_PATH_TWO = "./Data/iris_2.data";

        public static void Main(string[] args)
        {
            List<float[]> data = new();
            using (DataReader reader = new(DATA_PATH_TWO))
            {
                data = reader.ReadAll();
            }
            
            var splitData = DataProcessor.SplitData(data, 0.3m);

            List<float[]> trainingData = splitData.Item1;

            Neuron neuron = new SigmoidNeuron(trainingData[0].Length-1);


            int iterations = 0; int i = 0;
            while (iterations < 50)
            {
                if (i >= trainingData.Count) i = 0;

                float[] inputs = trainingData[i].SkipLast(1).ToArray();
                NeuronSupervisor.UpdateWeights(neuron, inputs, trainingData[i].Last(), 0.1f);

                i++;
                iterations++;
            }

            List<float[]> testingData = splitData.Item2;
            int correctPredictions = 0;
            float e = 0f;
            foreach (float[] values in testingData)
            {
                float calculated = neuron.Compute(values.SkipLast(1).ToArray());
                float expected = values.Last();
                e += MathF.Pow(calculated - expected, 2);
                if (MathF.Round(calculated) == expected) correctPredictions++;
            }
            Console.WriteLine($"{correctPredictions}/{testingData.Count}");
            Console.WriteLine($"{100.0f*correctPredictions/testingData.Count}%");
            Console.WriteLine($"{e/testingData.Count}");
        }
    }
}
