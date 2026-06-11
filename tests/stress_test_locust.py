import random
from locust import HttpUser, task, between

class SluiceUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def chat_request(self):
        """Simulates a small chat request (Low Priority)."""
        payload = {
            "model": "qwen-3.6",
            "messages": [{"role": "user", "content": "Tell me a short joke."}],
            "max_tokens": 50,
            "required_ctx": 2048 # Declares small context
        }
        with self.client.post("/v1/chat/completions", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # 503 is an expected 'Blocked' state in our architecture
                response.failure("Token Bank Full (Expected under load)")
            else:
                response.failure(f"Unexpected error: {response.status_code}")

    @task(1)
    def coding_request(self):
        """Simulates a large coding request (High Priority)."""
        payload = {
            "model": "qwen-3.6",
            "messages": [{"role": "user", "content": "Refactor this complex 32k line file..."}],
            "max_tokens": 500,
            "required_ctx": 32768 # Declares large context
        }
        # We use a virtual context URL for this one to test routing
        with self.client.post("/v1/ctx/32768/chat/completions", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Large request failed: {response.status_code}")

    @task(1)
    def check_metrics(self):
        """Simulates a monitoring tool scraping metrics."""
        self.client.get("/metrics")
