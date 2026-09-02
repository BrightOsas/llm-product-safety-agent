"""
agent.py

The agent loop: given a user question, let the LLM decide which tool(s)
from tools.py to call (via OpenAI function calling), execute them, and
generate a final grounded answer.
"""

import json
import os

from openai import OpenAI

from tools import RecallTools, TOOL_SCHEMAS
SYSTEM_PROMPT = """
You are "Is My Stuff Safe?" — a friendly household product safety
assistant backed by official CPSC recall data.

Your knowledge base currently covers ONLY these 10 product categories:
stroller, car seat, crib, baby swing, high chair, space heater,
pressure cooker, air fryer, coffee maker, battery charger.

If asked what you cover, or about ANY category NOT in this list (e.g.
toys, furniture, electronics, microwaves, bicycles, TVs, etc.), you MUST:
1. Clearly state that you don't have data for that category.
2. Immediately list the 10 categories you DO have data for, so the user
   knows what they can actually ask about instead.

Never claim "no recalls found" for a category you don't have data for —
that falsely implies you checked and found nothing, when you never had
the data to check in the first place. Being upfront about the boundary
of your own knowledge is more important than sounding comprehensive.

You have three tools:
- search_recalls: search the recall database by keyword/category. Use
  list_all=true when the user wants a full/complete list rather than
  just top matches.
- check_my_product: check whether a specific product the user describes
  matches a known recall
- compare_brand_safety: compare recall history between two brands

Use tools whenever the user asks about a specific product, brand, or
category within your 10 supported categories — do not answer from
general knowledge. If a question needs more than one tool (e.g. "is my
stroller safe, and is this brand generally trustworthy?"), call multiple
tools before answering.

When you give a final answer:
- Be clear about whether a match was found or not.
- Include the specific model number(s) when available.
- Include the hazard, date, and remedy when a match exists.
- Keep a calm, practical tone — never alarmist.
- If no data supports an answer, say so plainly instead of guessing.
""".strip()

MAX_TOOL_ROUNDS = 4


class RecallAgent:
    def __init__(self, recall_tools: RecallTools, llm_client: OpenAI | None = None, model: str | None = None):
        self.tools = recall_tools
        self.client = llm_client or OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self._dispatch = {
            "search_recalls": self.tools.search_recalls,
            "check_my_product": self.tools.check_my_product,
            "compare_brand_safety": self.tools.compare_brand_safety,
        }

    def _call_tool(self, name: str, arguments: dict):
        func = self._dispatch.get(name)
        if func is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return func(**arguments)
        except Exception as e:
            return {"error": str(e)}

    def ask(self, question: str, verbose: bool = False) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_calls_made = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return {
                    "answer": msg.content,
                    "tool_calls": tool_calls_made,
                }

            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"[tool call] {tc.function.name}({args})")

                result = self._call_tool(tc.function.name, args)
                tool_calls_made.append({"name": tc.function.name, "arguments": args})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

        final = self.client.chat.completions.create(model=self.model, messages=messages)
        return {
            "answer": final.choices[0].message.content,
            "tool_calls": tool_calls_made,
        }