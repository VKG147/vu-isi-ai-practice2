using System;

namespace AI.Practice1
{
    /// <summary>
    /// A single artifical neuron
    /// </summary>
    public abstract class Neuron
    {
        protected readonly Func<float, float> _actFunc;
        protected float[] _weights;

        public float[] Weights 
        {
            get { return _weights; }
            set 
            { 
                if (value.Length != _weights.Length)
                    throw new Exception($"Length of {nameof(Weights)} is immutable.");
                _weights = value;
            }
        }

        /// <summary>
        /// Creates a new Neuron
        /// </summary>
        /// <param name="actFunc">Activation function</param>
        /// <param name="weights">Initial weights</param>
        /// <param name="inputCount">Number of inputs</param>
        public Neuron(Func<float, float> actFunc, float[] weights)
        {
            _actFunc = actFunc;
            _weights = weights;
        }

        /// <summary>
        /// Creates a new Neuron
        /// </summary>
        /// <param name="actFunc">Activation function</param>
        /// <param name="weightCount">Number of weights</param>
        public Neuron(Func<float, float> actFunc, int weightCount)
        {
            _actFunc = actFunc;
            if (weightCount < 1)
                throw new Exception($"Value of {nameof(weightCount)} must be greater than 0.");
            _weights = new float[weightCount];
        }

        /// <summary>
        /// Calculates the neuron's output based on given inputs
        /// </summary>
        /// <returns>Neuron's output</returns>
        public float Compute(float[] inputs)
        {
            if (inputs.Length != _weights.Length)
                throw new Exception($"{nameof(inputs)} length must be equal to length of {nameof(Weights)}.");

            float a = 0f;
            for (int i = 0; i < inputs.Length; ++i)
            {
                a += inputs[i] * _weights[i];
            }
            return _actFunc(a);
        }

        /// <summary>
        /// Calculates the neuron's output based on given weights and inputs
        /// </summary>
        /// <returns>yNeuron's output</returns>
        public float Compute(float[] inputs, float[] weights)
        {
            if (inputs.Length != weights.Length)
                throw new Exception($"{nameof(inputs)} length must be equal to length of {nameof(weights)}.");

            float a = 0f;
            for (int i = 0; i < inputs.Length; ++i)
            {
                a += inputs[i] * weights[i];
            }
            return _actFunc(a);
        }

        public float[] ComputeWeightGradient(float[] inputs, float expectedOutput)
        {
            return ComputeWeightGradient(inputs, _weights, expectedOutput);
        }

        public virtual float[] ComputeWeightGradient(float[] inputs, float[] weights, float expectedOutput)
        {
            if (inputs.Length != _weights.Length)
                throw new Exception($"{nameof(inputs)} length must be equal to length of {nameof(weights)}.");

            float computedOutput = Compute(inputs, weights);

            float[] gradient = new float[weights.Length];
            for (int i = 0; i < gradient.Length; ++i)
            {
                gradient[i] = (expectedOutput - computedOutput) * (-inputs[i]);
            }

            return gradient;
        }
    }
}
