from openai import OpenAI, RateLimitError, OpenAIError
import sys
import os
import json

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from KEYS import openai_api_key

# Extract subjective values from text, summary pairs
def extract_subjective(text, summary):
        client = OpenAI(api_key = openai_api_key)

        prompt_system = (
            "You are an evaluation assistant. Please rate the following text on five criteria: "
            "information similarity, grammatical correctness, cohesion, conciseness, and provide an overall score.\n"
            "Output your response strictly as a valid JSON object with numeric scores for each of these five attributes "
            "on a scale from 0 to 1.\n"
            "Example format:\n"
            '{\n'
            '  "information_similarity": 0.95,\n'
            '  "grammatical_correctness": 0.90,\n'
            '  "cohesion": 0.85,\n'
            '  "conciseness": 0.80,\n'
            '  "overall_score": 0.88\n'
            '}'
        )

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": f"Input: {text}\nSummary: {summary}\n\nExtract the following subjective measures as a JSON object with keys 'information_similarity', 'grammatical_correctness', 'cohesion', 'conciseness', and 'overall_score' (scale of 0-1)."}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
            raw_output = response.choices[0].message.content.strip()
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            return None
            # Read retry-after header here if possible, or implement retry logic
        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            print(f"General error: {e}")
            return None

        try:
            result = json.loads(raw_output)
            # Basic validation of keys and value ranges
            keys = ['information_similarity', 'grammatical_correctness', 'cohesion', 'conciseness', 'overall_score']
            if all(k in result and isinstance(result[k], (int, float)) and 0 <= result[k] <= 1 for k in keys):
                return result
        except json.JSONDecodeError:
            pass