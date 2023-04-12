using BertTest;
using System;
using System.Text.Json;

var model = new Bert(
    Directory.GetCurrentDirectory() + "\\Assets\\Vocabulary\\vocab.txt",
    Directory.GetCurrentDirectory() + "\\Assets\\Model\\bertsquad-10.onnx"
    );

var (tokens, probability) = model.Predict("Jim is walking through the woods.", "What is his name?");

Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));
