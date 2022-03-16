using AI.Practice1;

namespace AI.Practice2
{
    public static class NeuronSupervisor
    {
        // based on given input
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
