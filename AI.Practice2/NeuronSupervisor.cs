using AI.Practice1;
using System;

namespace AI.Practice2
{
    public static class NeuronSupervisor
    {
        public static void InitializeNeuronWeights(Neuron neuron, Random random, float minWeight = -10f, float maxWeight = 10f)
        {
            for (int i = 0; i < neuron.Weights.Length; ++i)
            {
                neuron.Weights[i] = (float)random.NextDouble() * (maxWeight - minWeight) + minWeight;
            }
        }

        public static void UpdateWeights(Neuron neuron, float[] inputs, float expectedOutput, float learningRate)
        {
            float[] weightGradient = neuron.ComputeWeightGradient(inputs, expectedOutput);
            for (int i = 0; i < neuron.Weights.Length; ++i)
            {
                neuron.Weights[i] -= learningRate * weightGradient[i];
            }
        }
    }
}
