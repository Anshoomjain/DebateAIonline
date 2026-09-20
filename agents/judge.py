"""
Judge Agent - Balanced Synthesizer
===================================

Uses gemini-1.5-pro for better reasoning on synthesis.
"""

import os
import re
import time
from typing import List, Optional
import google.generativeai as genai

from core.interfaces import BaseAgent, Document, DebateState


genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
# Using flash for judge too to stay within free limits
_model = genai.GenerativeModel("gemini-3.6-flash")


class JudgeAgent(BaseAgent):
    """Judge Agent synthesizes Pro and Con arguments."""

    def __init__(self):
        super().__init__(
            name="judge",
            model="gemini-1.5-flash",
            role="Balanced synthesizer and judge"
        )
        print(f"✓ JudgeAgent initialized (Gemini)")

    def generate(self, query: str, context: List[Document],
                 debate_state: Optional[DebateState] = None) -> str:

        if not debate_state or not debate_state.rounds:
            return "[Error: No debate to judge]"

        pro_args = self._get_arguments(debate_state, 'pro')
        con_args = self._get_arguments(debate_state, 'con')
        context_text = self.format_context(context)
        prompt = self._build_prompt(query, context_text, pro_args, con_args)

        for attempt in range(3):
            try:
                response = _model.generate_content(
                    self.system_prompt + "\n\n" + prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=800,
                    )
                )
                verdict_text = response.text.strip()

                trust_score = self._calculate_trust_score(
                    pro_args, con_args, context
                )

                if debate_state:
                    debate_state.trust_score = trust_score
                    debate_state.verdict = self._extract_verdict(verdict_text)

                return (
                    f"{verdict_text}\n\n"
                    f"{'='*60}\nTRUST SCORE: {trust_score:.1f}%\n{'='*60}"
                )

            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return f"[Error: Judge Agent failed: {e}]"

    def _build_prompt(self, query: str, context: str,
                      pro_args: str, con_args: str) -> str:
        return f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

PRO AGENT'S ARGUMENTS:
{pro_args}

CON AGENT'S ARGUMENTS:
{con_args}

YOUR TASK: Provide a balanced final verdict.
Start your response with: VERDICT: FAVORABLE / UNFAVORABLE / UNCERTAIN

RESPONSE:"""

    def _get_arguments(self, debate_state: DebateState, agent_name: str) -> str:
        all_args = []
        for round_data in debate_state.rounds:
            if agent_name in round_data:
                all_args.append(round_data[agent_name]['content'])
        return "\n\n".join(all_args) if all_args else "No arguments"

    def _extract_verdict(self, text: str) -> str:
        if text.startswith("[Error:"):
            return "UNCERTAIN"
        text_upper = text.upper()
        match = re.search(
            r'VERDICT:\s*(FAVORABLE|UNFAVORABLE|UNCERTAIN)', text_upper
        )
        if match:
            return match.group(1)
        if 'UNFAVORABLE' in text_upper:
            return 'UNFAVORABLE'
        if 'FAVORABLE' in text_upper:
            return 'FAVORABLE'
        return 'UNCERTAIN'

    def _calculate_trust_score(self, pro_args: str, con_args: str,
                                context: List[Document]) -> float:
        citation_rate = self._calc_citation_rate(pro_args, con_args)
        balance = self._calc_balance(pro_args, con_args)
        recency = 0.8  # Default moderate recency

        score = citation_rate * 0.5 + balance * 0.3 + recency * 0.2
        return min(100.0, max(0.0, score * 100))

    def _calc_citation_rate(self, pro: str, con: str) -> float:
        combined = pro + " " + con
        citations = len(re.findall(r'\[Source:', combined, re.IGNORECASE))
        sentences = len(re.findall(r'[.!?]+', combined))
        if sentences == 0:
            return 0.0
        return min(1.0, citations / sentences)

    def _calc_balance(self, pro: str, con: str) -> float:
        pw, cw = len(pro.split()), len(con.split())
        if pw + cw == 0:
            return 0.0
        ratio = pw / (pw + cw)
        return 1.0 - abs(ratio - 0.5) * 2
