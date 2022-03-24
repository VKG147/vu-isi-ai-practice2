namespace AI.Practice1
{
    /// <summary>
    /// A type of Neuron with a binary stop activation function
    /// </summary>
    public class BinaryStopNeuron : Neuron
    {
        private static float BinaryStopFunc(float a)
        {
            return a >= 0 ? 1 : 0;
        }

        public BinaryStopNeuron(float[] weights) : base(BinaryStopFunc, weights) { }
        public BinaryStopNeuron(int weightCount) : base(BinaryStopFunc, weightCount) { }
    }
}
