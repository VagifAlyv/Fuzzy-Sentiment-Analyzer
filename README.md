# Fuzzy Sentiment Analyzer

A sophisticated sentiment analysis tool that combines VADER sentiment analysis with fuzzy logic and sarcasm detection capabilities. This analyzer provides detailed sentiment analysis with confidence scores and explanations for its decisions.

## Features

- **Fuzzy Logic Integration**: Uses fuzzy sets and rules for more nuanced sentiment analysis
- **Sarcasm Detection**: Identifies sarcastic content through various patterns and markers
- **Detailed Analysis**: Provides confidence scores, explanations, and token-level analysis
- **Emoji Support**: Includes sentiment analysis for common emojis
- **Negation Handling**: Properly handles negations and their scope
- **Intensifier Recognition**: Recognizes and processes intensifying words
- **Neutral Phrase Detection**: Identifies explicitly neutral statements

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fuzzy-sentiment-analyzer.git
cd fuzzy-sentiment-analyzer
```

2. Install required dependencies:
```bash
pip install nltk
```

3. The code will automatically download the required NLTK data (VADER lexicon) on first run.

## Usage

```python
from SentimentAnalyzer import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze text
text = "This product is absolutely amazing!"
result = analyzer.analyze_sentiment(text)

# Print results
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
print(f"Explanation: {result['explanation']}")
```

## Example Output

```python
{
    "sentiment": "very positive",
    "confidence": 0.85,
    "confidence_level": "high",
    "explanation": "Strong positive sentiment with intensifier",
    "is_sarcastic": false,
    "tokens": [
        {
            "token": "absolutely",
            "impact": 0.8,
            "reason": "intensifier (1.3x)"
        },
        {
            "token": "amazing",
            "impact": 0.9,
            "reason": "positive term (score: 0.90)"
        }
    ]
}
```

## Key Components

- **SentimentAnalyzer**: Main class for sentiment analysis
- **FuzzyKnowledgeBase**: Manages fuzzy sets and rules
- **Fuzzifier**: Converts crisp inputs to fuzzy values
- **InferenceEngine**: Applies fuzzy rules for analysis
- **Defuzzifier**: Converts fuzzy outputs to crisp results

## Configuration

The analyzer can be customized by adjusting various parameters in the `SentimentAnalyzer` class:

- Confidence thresholds
- Sentiment thresholds
- Negation parameters
- Intensifier multipliers
- Sarcasm detection patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VADER Sentiment Analysis
- NLTK library
- Fuzzy Logic principles

## Contact

For questions or suggestions, please open an issue in the GitHub repository.
