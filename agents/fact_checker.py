"""
Fact-Checker Agent - Validates Claims Against Sources
======================================================

"""

import os
import re
import time
from typing import List, Optional
import google.generativeai as genai

from core.interfaces import BaseAgent, Document, DebateState


genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
_model = genai.GenerativeModel("gemini-1.5-flash")


class FactCheckerAgent(BaseAgent):
    """Fact-Checker Agent verifies claims against source documents."""

    def __init__(self):
        super().__init__(
            name="fact_checker",
            model="gemini-1.5-flash",
            role="Claim validator and fact-checker"
        )
        print(f"✓ FactCheckerAgent initialized (Gemini)")

    def generate(self, query: str, context: List[Document],
                 debate_state: Optional[DebateState] = None) -> str:

        if not debate_state or not debate_state.rounds:
            return "[Error: No arguments to fact-check]"

        pro_args = self._get_arguments(debate_state, 'pro')
        con_args = self._get_arguments(debate_state, 'con')
        all_claims = self._extract_claims(pro_args, con_args)

        if not all_claims:
            return "No specific cited claims found to verify."

        context_text = self.format_context(context)
        prompt = self._build_prompt(all_claims, context_text)

        for attempt in range(3):
            try:
                response = _model.generate_content(
                    self.system_prompt + "\n\n" + prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=600,
                    )
                )
                report = response.text.strip()
                pass_rate = self._calculate_pass_rate(report)

                summary = (
                    f"\n\n{'='*60}\nFACT-CHECK SUMMARY\n{'='*60}\n"
                    f"Claims Checked: {len(all_claims)}\n"
                    f"Pass Rate: {pass_rate:.1f}%\n"
                    f"{'='*60}"
                )
                return report + summary

            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return f"[Error: Fact-Checker failed: {e}]"

    def _build_prompt(self, claims: List[str], context: str) -> str:
        claims_text = "\n".join(
            [f"{i+1}. {c}" for i, c in enumerate(claims)]
        )
        return f"""CONTEXT DOCUMENTS:
{context}

CLAIMS TO VERIFY:
{claims_text}

For each claim, state:
- Status: ✓ VERIFIED / ✗ UNVERIFIED / ⚠ PARTIALLY VERIFIED
- Evidence: cite the source document

RESPONSE:"""

    def _get_arguments(self, debate_state: DebateState, agent_name: str) -> str:
        all_args = []
        for round_data in debate_state.rounds:
            if agent_name in round_data:
                content = round_data[agent_name]['content']
                if not content.startswith("[Error:"):
                    all_args.append(content)
        return "\n\n".join(all_args)

    def _extract_claims(self, pro_args: str, con_args: str) -> List[str]:
        combined = pro_args + "\n\n" + con_args
        matches = re.findall(r'([^.!?\n]+\[Source:[^\]]+\])', combined)
        return [m.strip() for m in matches if len(m.strip()) > 20][:10]

    def _calculate_pass_rate(self, report: str) -> float:
        verified = report.count('✓ VERIFIED')
        unverified = report.count('✗ UNVERIFIED')
        partial = report.count('⚠ PARTIALLY VERIFIED')
        total = verified + unverified + partial
        if total == 0:
            return 50.0
        return ((verified + partial * 0.5) / total) * 100
