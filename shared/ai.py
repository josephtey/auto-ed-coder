from shared.prompts import label_prompt, score_prompt
from openai import OpenAI
import json


class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_interpretation(self, positive_samples, negative_samples):
        prompt = label_prompt(positive_samples, negative_samples)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    def score_interpretation(self, samples, attributes):
        prompt = score_prompt(samples, attributes)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return json.loads(response.choices[0].message.content)
