from shared.prompts import get_prompt_by_type, score_prompt
from openai import OpenAI
import json
import tiktoken


class OpenAIClient:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.encoder = tiktoken.encoding_for_model(model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.model = model

    def get_interpretation(self, positive_samples, negative_samples, prompt_type):
        prompt = get_prompt_by_type(positive_samples, negative_samples, prompt_type)

        response = self.client.chat.completions.create(
            model=self.model,
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

        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        self.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)

        return json.loads(response.choices[0].message.content)

    def score_interpretation(self, samples, attributes):
        prompt = score_prompt(samples, attributes)

        response = self.client.chat.completions.create(
            model=self.model,
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

        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        self.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)

        return json.loads(response.choices[0].message.content)

    def update_cost(self, prompt_tokens, completion_tokens):
        prompt_cost = (prompt_tokens / 1_000_000) * 5
        completion_cost = (completion_tokens / 1_000_000) * 15
        self.total_cost += prompt_cost + completion_cost

        print(f"Total cost: ${self.total_cost:.2f}")

    def get_total_tokens(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": f"${self.total_cost:.2f}",
        }
