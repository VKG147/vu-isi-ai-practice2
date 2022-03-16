using System;
using System.Collections.Generic;
using System.Linq;

namespace AI.Practice2
{
    public static class DataProcessor
    {
        private static readonly Random random = new();

        /// <summary>
        /// Splits the provided neuron data into two groups based on the splitRatio
        /// </summary>
        /// <param name="splitRatio">Value from 0 to 1, the ratio of elements in the first group</param>
        /// <returns>Two lists of neuron data</returns>
        public static Tuple<List<float[]>, List<float[]>> SplitData(List<float[]> data, decimal splitRatio)
        {
            data.Shuffle();
            int groupOneCount = (int)(splitRatio * data.Count);
            List<float[]> groupOne = data.GetRange(0, groupOneCount).ToList();
            List<float[]> groupTwo = data.GetRange(groupOne.Count, data.Count - groupOneCount);
            return new(groupOne, groupTwo);
        }

        private static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
