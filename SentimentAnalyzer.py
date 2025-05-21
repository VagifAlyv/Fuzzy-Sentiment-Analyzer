import re
from collections import defaultdict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize sentiment a
        
        nalyzer with VADER's lexicon but custom fuzzy logic.
        Added sarcasm detection capabilities.
        """
        # Initialize VADER just to access its lexicon
        self.vader = SentimentIntensityAnalyzer()
        # Extract the lexicon for our tokenized analysis
        self.lexicon = self.vader.lexicon
        
        # CONFIGURATION PARAMETERS - easy to edit for tweaking
        
        # Confidence threshold values
        self.CONFIDENCE_HIGH = 0.7    # Threshold for high confidence
        self.CONFIDENCE_MEDIUM = 0.5  # Threshold for medium confidence
        self.CONFIDENCE_LOW = 0.3     # Threshold for low confidence
        
        # Fuzzy sentiment threshold values
        self.POSITIVE_THRESHOLD = 0.15  # Minimum score to be considered positive
        self.NEGATIVE_THRESHOLD = -0.15 # Maximum score to be considered negative
        
        # Sentiment normalization factors
        self.NORM_FACTOR = 4.0  # Normalizing factor for VADER scores (VADER uses -4 to +4)
        
        # Negation parameters
        self.NEGATION_SCOPE = 2  # Reduce from 3 to prevent over-negation
        self.NEGATION_FACTOR = 0.7  # How strongly negation affects sentiment
        
        # Intensifier base multiplier
        self.INTENSIFIER_BASE = 1.2  # Default intensifier if not specified
        
        # Intensifiers with specific multipliers
        self.intensifiers = {
            'very': 1.3, 'really': 1.2, 'extremely': 1.4, 'absolutely': 1.5, 
            'truly': 1.2, 'completely': 1.3, 'totally': 1.2, 'highly': 1.2, 
            'especially': 1.1, 'particularly': 1.1, 'incredibly': 1.4,
            'super': 1.3, 'quite': 1.1, 'so': 1.2
        }
        
        # Negations that flip sentiment
        self.negations = {
            'not', "don't", "doesn't", "didn't", "won't", "wouldn't", 
            "can't", "cannot", "couldn't", "shouldn't", "isn't", "aren't",
            "haven't", "hasn't", "hadn't", "never", "no", "nowhere", "nobody",
            "nothing", "none"
        }
        
        # Emoji sentiments with scores
        self.emoji_sentiments = {
            ':)': 0.7, ':-)': 0.7, ':(': -0.7, ':-(': -0.7,
            ':-D': 1.0, ':D': 1.0, ':-P': 0.3, ':P': 0.3,
            '❤️': 1.0, '😊': 0.8, '😢': -0.7, '😭': -0.9,
            '👍': 0.8, '👎': -0.8, '🙁': -0.5, '☹️': -0.7,
            '🙄': -0.3,  # Eye roll - often used with sarcasm
            '😏': -0.1,  # Smirk - often associated with sarcasm
        }

        # Add neutral indicators to the class initialization
        self.neutral_indicators = {
            'neither', 'nor', 'neutral', 'neither good nor bad', 
            'neither positive nor negative', 'neither happy nor sad',
            'neither like nor dislike', 'neither love nor hate', 'it was not good or bad',
            'it was not positive or negative', 'it was not happy or sad',
            'it was not like or dislike', 'it was not love or hate', 'it was not good or bad',
            'it was not positive or negative', 'it was not happy or sad',
            'it was not like or dislike', 'it was not love or hate', 'it was not good or bad',
            'it was not positive or negative', 'it was not happy or sad',
            'it was not like or dislike', 'it was not love or hate',             
            'fine', 'okay', 'ok', 'alright', 'could be better',
            'not sure', 'unsure', 'meh', 'whatever', 'normal',
            'average', 'decent', 'moderate', 'fair', 'reasonable'
        }

        # Add intensity thresholds
        self.VERY_POSITIVE_THRESHOLD = 0.7
        self.VERY_NEGATIVE_THRESHOLD = -0.7

        # Update lexicon with additional terms
        self.lexicon.update({
            'deserve': 0.4,
            'flower': 0.3,
            'better': 0.2,  # Context-dependent
            'allowed': -0.2,  # For rhetorical questions
            'thanks': 0.6,   # Positive unless used sarcastically
            'wow': 0.5,      # Can be positive or sarcastic
            'brilliant': 0.8, # Can be positive or sarcastic
            'genius': 0.7,   # Can be positive or sarcastic
            'amazing': 0.8,  # Can be positive or sarcastic
            'perfect': 0.9,  # Can be positive or sarcastic
            'terrific': 0.8, # Can be positive or sarcastic
            'right': 0.3,    # Can be positive or sarcastic
            'great': 0.7,    # Can be positive or sarcastic
            'clever': 0.6,   # Can be positive or sarcastic
        })

        # Sarcasm markers and patterns
        self.sarcasm_markers = {
            'thanks for nothing', 'oh great', 'oh perfect', 'wow just wow',
            'just what i needed', 'just what we needed', 'just great', 
            'how wonderful', 'my hero', 'my favorite', 'you don\'t say',
            'excellent work', 'fantastic job', 'brilliant idea', 'genius plan',
            'simply amazing', 'really helpful', 'super helpful',
            'exactly what i wanted', 'exactly what we wanted',
            'nice going', 'nice work', 'nice job', 'good job', 'good work',
            'totally useful', 'so useful', 'so helpful', 'perfect timing',
            'sure, why not', 'sure, whatever', 'yeah right', 'sure thing',
            'sooo', 'suuure', 'riiight', 'yeaah', 'exactly what i was hoping for',
            'just peachy', 'fantastic idea', 'amazingly helpful', 'absolutely brilliant',
            'aren\'t you special', 'just fabulous', 'wooow', 'woooow', 'suuuper',
            'couldn\'t be happier', 'couldn\'t be better'
        }
        
        # Sarcastic punctuation patterns
        self.sarcastic_punct_patterns = [
            r'\.\.\.',           # Ellipsis
            r'!!+',              # Multiple exclamation marks
            r'\?!+',             # Question mark followed by exclamation(s)
            r'!+\?',             # Exclamation(s) followed by question mark
            r'\?{2,}',           # Multiple question marks
        ]
        
        # Keywords that might indicate sarcasm when combined with positive sentiment
        self.sarcasm_keywords = {
            'obviously', 'clearly', 'totally', 'absolutely', 'definitely', 
            'certainly', 'precisely', 'of course', 'by all means',
            'sure', 'right', 'yeah', 'great', 'lovely', 'fantastic',
            'brilliant', 'amazing', 'impressive', 'extraordinary', 'wonderful',
            'congrats', 'congratulations', 'bravo', 'nice job', 'well done'
        }
        
        # Exaggeration patterns that might indicate sarcasm
        self.exaggeration_patterns = [
            r'\b(so+|very+|really+|extremely+|absolutely+|completely+|totally+)\b',
            r'\b(amazing|awesome|fantastic|incredible|unbelievable|brilliant)\b'
        ]

    def tokenize(self, text):
        """
        Tokenize the input text into words, removing punctuation except for important markers.
        Also handles emojis as tokens.
        Returns a list of tokens.
        """
        # Convert to lowercase
        text = text.lower()
        
        
        # Replace contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        
        # Split on whitespace and punctuation, but keep exclamation and question marks as tokens
        tokens = []
        words = re.findall(r'\b\w+\b|[!?]', text)
        
        # Add words and punctuation
        for word in words:
            tokens.append(word)
        
            
        return tokens
    
    def detect_sarcasm(self, text, tokens, positive_score, negative_score):
        """
        Detect if a text is likely sarcastic based on various patterns and cues.
        Returns a tuple of (is_sarcastic, confidence, explanation)
        """
        text_lower = text.lower()
        sarcasm_score = 0.0
        explanations = []
        
        # Check for known sarcastic phrases
        for marker in self.sarcasm_markers:
            if marker in text_lower:
                sarcasm_score += 0.7
                explanations.append(f"Detected known sarcastic phrase: '{marker}'")
                break
                
        # Check for sarcastic punctuation
        for pattern in self.sarcastic_punct_patterns:
            if re.search(pattern, text):
                sarcasm_score += 0.3
                explanations.append(f"Detected sarcastic punctuation pattern: '{pattern}'")
                break
                
        # Check for exaggerated positive terms with low context positivity
        if positive_score > 0.6 and negative_score < 0.2:
            for pattern in self.exaggeration_patterns:
                if re.search(pattern, text_lower):
                    # Check if we also have contextual negativity markers
                    has_neg_context = False
                    for neg_word in ["but", "however", "though", "actually", "exactly", "right"]:
                        if neg_word in text_lower:
                            has_neg_context = True
                            break
                    
                    if has_neg_context:
                        sarcasm_score += 0.4
                        explanations.append("Exaggerated positive terms with negative context")
                    break
        
        # Check for overly positive sentiment with specific sarcasm keywords
        keyword_count = sum(1 for keyword in self.sarcasm_keywords if keyword in text_lower)
        if keyword_count >= 2 and positive_score > 0.6:
            sarcasm_score += 0.3 * min(keyword_count, 3)  # Cap at 3 keywords
            explanations.append(f"Multiple sarcasm keywords ({keyword_count}) with high positive sentiment")
        
        # Check for "thanks" combined with negative sentiment or negative terms
        if "thanks" in text_lower and (negative_score > 0.3 or 
                                       any(neg in text_lower for neg in ["nothing", "lot", "oh", "great", "right"])):
            sarcasm_score += 0.6
            explanations.append("'Thanks' combined with negative context")
        
        # Check for contrast between positive and negative sentiment (a sarcasm indicator)
        sentiment_contrast = abs(positive_score - negative_score)
        if positive_score > 0.5 and negative_score > 0.3 and sentiment_contrast < 0.4:
            sarcasm_score += 0.4
            explanations.append("Mixed positive and negative sentiment (likely sarcasm)")
        
        # Check for emphatic punctuation
        exclamation_count = text.count('!')
        if exclamation_count >= 2:
            sarcasm_score += 0.2
            explanations.append(f"Multiple exclamation marks ({exclamation_count})")
        
        # Final determination
        is_sarcastic = sarcasm_score >= 0.5
        confidence = min(1.0, sarcasm_score)
        
        return is_sarcastic, confidence, explanations

    def analyze_sentiment(self, text):
        tokens = self.tokenize(text)
        
        if not tokens:
            return self._create_result(0, 0, 0, 0, "neutral", "Empty text")

        # Initialize fuzzy variables
        positive_sum = 0.0
        negative_sum = 0.0
        token_sentiment = []
        negate = False
        negation_scope = 0
        intensify = 1.0

        # Fuzzy membership functions for sentiment strength
        def fuzzy_membership(score):
            if abs(score) < 0.1: return 0.0
            if abs(score) < 0.3: return 0.3
            if abs(score) < 0.5: return 0.6
            return 1.0

        # Process tokens
        for i, token in enumerate(tokens):
            token_info = {"token": token, "impact": 0, "positive": 0, "negative": 0, "reason": ""}
            
            # Handle negations and intensifiers
            if token in self.negations:
                negate = not negate
                negation_scope = self.NEGATION_SCOPE
                token_info["reason"] = "negation"
                token_sentiment.append(token_info)
                continue
            
            if token in self.intensifiers:
                intensify = self.intensifiers[token]
                token_info["reason"] = f"intensifier ({intensify}x)"
                token_sentiment.append(token_info)
                continue

            # Get sentiment value
            sentiment_value = 0
            if token in self.emoji_sentiments:
                sentiment_value = self.emoji_sentiments[token]
                token_info["reason"] = f"emoji (score: {sentiment_value:.2f})"
            elif token in self.lexicon:
                sentiment_value = self.lexicon[token] / self.NORM_FACTOR
                token_info["reason"] = f"{'positive' if sentiment_value > 0 else 'negative' if sentiment_value < 0 else 'neutral'} term (score: {sentiment_value:.2f})"
            else:
                token_info["reason"] = "unknown term (not in lexicon)"

            # Apply fuzzy rules
            if negate and negation_scope > 0:
                sentiment_value = -sentiment_value * self.NEGATION_FACTOR
                token_info["reason"] += " (negated)"
                negation_scope -= 1
                if negation_scope == 0:
                    negate = False

            if intensify > 1.0:
                sentiment_value *= intensify
                token_info["reason"] += f" (intensified {intensify:.1f}x)"
                intensify = 1.0

            # Update scores with fuzzy membership
            token_info["impact"] = round(sentiment_value, 2)
            if sentiment_value > 0:
                token_info["positive"] = fuzzy_membership(sentiment_value)
                positive_sum += token_info["positive"]
            elif sentiment_value < 0:
                token_info["negative"] = fuzzy_membership(abs(sentiment_value))
                negative_sum += token_info["negative"]

            token_sentiment.append(token_info)

        # Calculate final scores
        sentiment_words = sum(1 for t in token_sentiment if abs(t["impact"]) > 0.05)
        normalization_factor = max(1, sentiment_words) * 0.5
        
        positive_score = min(1.0, positive_sum / normalization_factor) if sentiment_words > 0 else 0.0
        negative_score = min(1.0, negative_sum / normalization_factor) if sentiment_words > 0 else 0.0
        net_score = positive_score - negative_score if sentiment_words > 0 else 0.0

        # Check for sarcasm
        is_sarcastic, sarcasm_confidence, sarcasm_reasons = self.detect_sarcasm(
            text, tokens, positive_score, negative_score
        )
        
        # Check for explicit neutral phrases
        text_lower = text.lower()
        is_explicitly_neutral = any(indicator in text_lower for indicator in self.neutral_indicators)
        
        # Enhanced sentiment determination with intensity levels and sarcasm awareness
        if is_explicitly_neutral or (abs(net_score) < 0.1 and not is_sarcastic):
            sentiment = "neutral"
            if is_explicitly_neutral:
                explanation = "Explicitly states a neutral position"
            elif sentiment_words == 0:
                explanation = "No sentiment-bearing words found"
            elif positive_score > 0 and negative_score > 0:
                explanation = f"Mixed sentiments (pos: {positive_score:.2f}, neg: {negative_score:.2f})"
            else:
                explanation = f"Weak sentiment (strength: {abs(net_score):.2f})"
        else:
            # Determine sentiment intensity using original scores
            if net_score >= self.VERY_POSITIVE_THRESHOLD:
                sentiment = "very positive"
            elif net_score <= self.VERY_NEGATIVE_THRESHOLD:
                sentiment = "very negative"
            elif net_score > 0.1:
                if net_score < 0.3:
                    sentiment = "slightly positive"
                else:
                    sentiment = "positive"
            elif net_score < -0.1:
                if net_score > -0.3:
                    sentiment = "slightly negative"
                else:
                    sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Add sarcasm note if detected
            if is_sarcastic:
                sentiment += " (potentially sarcastic)"
                explanation = f"Sarcasm detected with {sarcasm_confidence:.2f} confidence"
            else:
                explanation = ""

        # Initialize confidence
        confidence = 0.0

        # Adjust confidence calculation for better accuracy, considering sarcasm
        if sentiment_words > 0:
            raw_confidence = (positive_score + negative_score) * min(1.0, sentiment_words / 3.0)
            
            # Adjust confidence based on sentiment and sarcasm
            if is_sarcastic:
                # Sarcasm detection typically increases uncertainty
                confidence = raw_confidence * (0.7 + 0.3 * sarcasm_confidence)
            elif "very" in sentiment:
                confidence = raw_confidence * 1.2  # Increase confidence for strong sentiments
            elif sentiment == "neutral":
                # Reduce confidence for neutral cases with mixed sentiments
                if positive_score > 0 and negative_score > 0:
                    ratio = min(positive_score, negative_score) / max(positive_score, negative_score)
                    confidence = raw_confidence * (1 - ratio * 0.7)
                else:
                    confidence = raw_confidence * 0.8
            else:
                confidence = raw_confidence

        # Determine confidence level
        confidence_level = "high" if confidence >= self.CONFIDENCE_HIGH else \
                         "medium" if confidence >= self.CONFIDENCE_MEDIUM else \
                         "low" if confidence >= self.CONFIDENCE_LOW else "very low"

        # Add special handling for rhetorical questions
        if '?' in text and any(word in text.lower() for word in ['why', 'how', 'what']):
            # Adjust sentiment based on context
            if 'allowed' in text.lower() or 'exist' in text.lower():
                negative_score += 0.4

        # Add sarcasm detection for individual tokens
        for token_info in token_sentiment:
            token = token_info['token']
            # Check if token is part of any sarcastic phrases
            is_sarcastic_token = False
            for marker in self.sarcasm_markers:
                if token in marker.split():
                    is_sarcastic_token = True
                    break
            
            # Check if token is a sarcasm keyword
            if token in self.sarcasm_keywords:
                is_sarcastic_token = True
            
            token_info['is_sarcastic'] = is_sarcastic_token

        result = {
            "original_positive_score": round(positive_score, 2),
            "original_negative_score": round(negative_score, 2),
            "original_net_score": round(net_score, 2),
            "adjusted_positive_score": round(positive_score, 2),
            "adjusted_negative_score": round(negative_score, 2),
            "adjusted_net_score": round(net_score, 2),
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level,
            "sentiment": sentiment,
            "explanation": explanation,
            "tokens": token_sentiment,
            "is_sarcastic": is_sarcastic,
            "sarcasm_confidence": round(sarcasm_confidence, 2) if is_sarcastic else 0.0,
            "sarcasm_reasons": sarcasm_reasons if is_sarcastic else []
        }
        
        return result

class   FuzzyKnowledgeBase:
    def __init__(self):
        # Define fuzzy sets for sentiment and sarcasm
        self.sentiment_sets = {
            "negative": lambda x: max(0, min(1, -x)),  # x in [-1, 0]
            "neutral": lambda x: max(0, 1 - abs(x)),   # x in [-1, 1]
            "positive": lambda x: max(0, min(1, x)),   # x in [0, 1]
        }
        self.sarcasm_sets = {
            "low": lambda x: max(0, 1 - x),
            "high": lambda x: max(0, x),
        }
        # Example fuzzy rules
        self.rules = [
            # (sentiment, sarcasm) => output
            (("positive", "high"), "sarcastic_positive"),
            (("negative", "high"), "sarcastic_negative"),
            (("positive", "low"), "genuine_positive"),
            (("negative", "low"), "genuine_negative"),
            (("neutral", "low"), "neutral"),
            (("neutral", "high"), "sarcastic_neutral"),
        ]

class Fuzzifier:
    def __init__(self, lexicon, emoji_sentiments):
        self.lexicon = lexicon
        self.emoji_sentiments = emoji_sentiments

    def fuzzify(self, text):
        # Tokenize and compute a raw sentiment score in [-1, 1]
        # (No hard thresholding, just sum and normalize)
        tokens = re.findall(r'\b\w+\b|[!?]', text.lower())
        score = 0.0
        count = 0
        for token in tokens:
            if token in self.emoji_sentiments:
                score += self.emoji_sentiments[token]
                count += 1
            elif token in self.lexicon:
                score += self.lexicon[token] / 4.0
                count += 1
        sentiment = score / count if count else 0.0
        # Fuzzify sarcasm as a degree (e.g., based on markers, punctuation, etc.)
        sarcasm = min(1.0, text.count('!') / 3.0)  # Example: more exclamations = more sarcasm
        return {"sentiment": sentiment, "sarcasm": sarcasm}

class InferenceEngine:
    def __init__(self, kb):
        self.kb = kb

    def infer(self, fuzzy_inputs):
        # Compute membership degrees
        sentiment_degrees = {k: f(fuzzy_inputs["sentiment"]) for k, f in self.kb.sentiment_sets.items()}
        sarcasm_degrees = {k: f(fuzzy_inputs["sarcasm"]) for k, f in self.kb.sarcasm_sets.items()}
        # Apply rules and aggregate outputs
        output_degrees = defaultdict(float)
        for (sentiment_label, sarcasm_label), output_label in self.kb.rules:
            degree = min(sentiment_degrees[sentiment_label], sarcasm_degrees[sarcasm_label])
            output_degrees[output_label] = max(output_degrees[output_label], degree)
        return output_degrees

class Defuzzifier:
    def defuzzify(self, output_degrees):
        # Pick the label with the highest degree
        if not output_degrees:
            return "neutral"
        return max(output_degrees.items(), key=lambda x: x[1])[0]

# Main fuzzy sentiment analyzer
class FuzzySentimentAnalyzer:
    def __init__(self):
        vader = SentimentIntensityAnalyzer()
        lexicon = vader.lexicon
        emoji_sentiments = {
            ':)': 0.7, ':(': -0.7, ':D': 1.0, '❤️': 1.0, '😢': -0.7, '👍': 0.8, '👎': -0.8
        }
        self.kb = FuzzyKnowledgeBase()
        self.fuzzifier = Fuzzifier(lexicon, emoji_sentiments)
        self.inference = InferenceEngine(self.kb)
        self.defuzzifier = Defuzzifier()

    def analyze(self, text):
        fuzzy_inputs = self.fuzzifier.fuzzify(text)
        output_degrees = self.inference.infer(fuzzy_inputs)
        result = self.defuzzifier.defuzzify(output_degrees)
        return {
            "fuzzy_inputs": fuzzy_inputs,
            "output_degrees": output_degrees,
            "result": result
        }

# Example usage
if __name__ == "__main__":
    analyzer = FuzzySentimentAnalyzer()
    sentences = [
        "Thanks for nothing!!!",
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "Well, that was helpful...",
        "I'm not sure how I feel about this.",
        "Great job!",
        "Could be better.",
        "Wow, just wow!!!",
        "It's okay, I guess.",
        "You did a fantastic job!!!",
        'You deserve flowers for that'
    ]

    for idx, sentence in enumerate(sentences, 1):
        result = analyzer.analyze(sentence)
        print(f"\n=== Fuzzy Sentiment Analysis Result #{idx} ===")
        print(f"Input sentence: {sentence}")
        print(f"Fuzzified values:")
        print(f"  - Sentiment degree: {result['fuzzy_inputs']['sentiment']:.2f}")
        print(f"  - Sarcasm degree:   {result['fuzzy_inputs']['sarcasm']:.2f}")
        print("\nFuzzy output degrees:")
        for label, degree in result['output_degrees'].items():
            print(f"  - {label}: {degree:.2f}")
        print(f"\nFinal sentiment classification: {result['result'].replace('_', ' ').capitalize()}")