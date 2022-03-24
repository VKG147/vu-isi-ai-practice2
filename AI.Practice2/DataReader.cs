using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AI.Practice2
{
    public class DataReader : IDisposable
    {
        private readonly StreamReader _fileReader;

        public DataReader(string path)
        {
            _fileReader = new StreamReader(path);
        }

        public List<float[]> ReadAll()
        {
            List<float[]> data = new();
            int expectedValuesCount = -1;
            while (!_fileReader.EndOfStream)
            {
                string line = _fileReader.ReadLine();
                float[] values = line.Split(',').Select(v => float.Parse(v)).ToArray();

                if (expectedValuesCount == -1)
                    expectedValuesCount = values.Length;
                else if (expectedValuesCount != values.Length)
                    throw new Exception($"Unexpected number of values encountered at line {values.Length + 1}.");

                data.Add(values);
            }
            return data;
        }

        public void Dispose()
        {
            _fileReader.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
