using System;

namespace AI.Practice1
{
    /// <summary>
    /// A type of Neuron with a sigmoid activation function
    /// </summary>
    public class SigmoidNeuron : Neuron
    {
        private static float SigmoidFunc(float a)
        {
            return 1 / (1 + MathF.Exp(-a));
        }

        public SigmoidNeuron(float[] weights) : base(SigmoidFunc, weights) { }
        public SigmoidNeuron(int weightCount) : base(SigmoidFunc, weightCount) { }
    }
}
