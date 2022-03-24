using System;
using System.Collections.Generic;
using System.Linq;

namespace AI.Practice2
{
    public class DataProcessor
    {
        private readonly Random _random;

        public DataProcessor() { _random = new(); }
        public DataProcessor(Random random) { _random = random; }

        /// <summary>
        /// Splits the provided neuron data into two groups based on the splitRatio
        /// </summary>
        /// <param name="splitRatio">Value from 0 to 1, the ratio of elements in the first group</param>
        /// <returns>Two lists of neuron data</returns>
        public (List<float[]> trainingData, List<float[]> testingData) SplitData(List<float[]> data, decimal splitRatio)
        {
            data.Shuffle(_random);
            int groupOneCount = (int)(splitRatio * data.Count);
            List<float[]> groupOne = data.GetRange(0, groupOneCount).ToList();
            List<float[]> groupTwo = data.GetRange(groupOne.Count, data.Count - groupOneCount);
            return new(groupOne, groupTwo);
        }

        public static List<float[]> PrependBiasInputs(List<float[]> data)
        {
            for (int i = 0; i < data.Count; ++i)
            {
                float[] inputs = new float[data[i].Length + 1];
                inputs[0] = 1;
                Array.Copy(data[i], 0, inputs, 1, data[i].Length);
                data[i] = inputs;
            }
            return data;
        }
    }
}
