
from locust import HttpUser, task, between
import random

EMAIL_BODIES = [
    "I have a billing issue with my invoice #12345. The amount charged was incorrect.",
    "My account is locked and I cannot log in. Please help reset my password.",
    "The technical system is down and I cannot access my data. This is urgent.",
    "I would like to request a refund for my cancelled subscription.",
    "My order has not been delivered. Tracking shows it is stuck at the warehouse.",
    "I have a general inquiry about your service pricing and features.",
]

class EmailUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def process_email(self):
        self.client.post("/emails/process", json={
            "sender": f"user{random.randint(1,1000)}@test.com",
            "subject": "Support Request",
            "body": random.choice(EMAIL_BODIES),
        }, timeout=30)
