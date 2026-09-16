"""
Language Agent — SignBridge V2
================================
Takes raw detected letters/words from the Vision pipeline (MediaPipe + 
trained model) and converts them into natural, grammatically correct
sentences using an LLM — instead of simple word-by-word autocomplete.

This is "Agent 2" in the multi-agent architecture:
  Agent 1 (Vision)   -> detects letters/words from hand signs
  Agent 2 (Language)  -> THIS FILE: refines into natural sentence
  Agent 3 (Animation) -> converts hearing person's reply back to ASL

Usage:
    from language_agent import LanguageAgent

    agent = LanguageAgent()
    sentence = agent.refine("h e l l o how are y o u")
    print(sentence)   # "Hello, how are you?"
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LanguageAgent:
    def __init__(self, model="openai/gpt-oss-20b"):
        """
        Initialize the Language Agent with a Groq client.
        Uses the same GROQ_API_KEY setup as MindEase.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file:\n"
                "GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def refine(self, raw_text: str) -> str:
        """
        Takes raw, possibly fragmented letters/words detected from
        sign language input, and returns a natural, grammatically
        correct sentence.

        Example:
            Input:  "hello how are you"
            Output: "Hello! How are you?"
        """
        if not raw_text or not raw_text.strip():
            return ""

        system_prompt = (
            "You are a language assistant helping convert fingerspelled "
            "ASL letters/words into natural, grammatically correct English "
            "sentences. The input comes from a real-time hand-sign detector "
            "and may be incomplete or garbled — some letters may be missing "
            "or misdetected (e.g. 'i wan o hme' likely means "
            "'I want to go home'), especially from learners who are not yet "
            "fluent in ASL. Your job is to:\n"
            "1. Infer the most likely INTENDED sentence, filling in missing "
            "letters/words where the intent is reasonably clear\n"
            "2. Add correct capitalization and punctuation\n"
            "3. Stay close to the words actually present — do not invent "
            "unrelated content or change the topic\n"
            "4. If the input is too ambiguous to confidently infer intent, "
            "return it with just grammar/punctuation cleanup instead of guessing\n"
            "5. Keep it concise — output ONLY the corrected sentence, "
            "nothing else (no explanations, no quotes)"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.3,  # low temperature = more consistent, less creative
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            # Fail gracefully — return the raw text if the LLM call fails
            # (e.g. network issue), so the app doesn't break entirely.
            print(f"[LanguageAgent] Error refining text: {e}")
            return raw_text


# ── Quick standalone test ────────────────────────────────────────────────
if __name__ == "__main__":
    agent = LanguageAgent()

    test_inputs = [
        "hello how are you",
        "i need help",
        "thank you very much",
        "nice to meet you",
    ]

    print("Testing Language Agent (raw -> refined):\n")
    for raw in test_inputs:
        refined = agent.refine(raw)
        print(f"  '{raw}'  ->  '{refined}'")