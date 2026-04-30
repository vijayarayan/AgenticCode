"""
Minimal ReAct agent — runnable demo.

Run it:
    python react_agent.py

By default it uses a MOCK LLM that returns scripted ReAct-formatted responses
so you can watch the loop work without any API key.

To use a real LLM (Anthropic), see the bottom of this file.
"""

import re


# ---------- The agent loop ----------

def run_agent(query, tools, llm, max_steps=5, verbose=True):
    tool_descriptions = "\n".join(
        f"- {name}: {fn.__doc__ or 'no description'}"
        for name, fn in tools.items()
    )

    system_prompt = f"""You are a ReAct agent. Solve the user's question by reasoning step by step.

Available tools:
{tool_descriptions}

Use this exact format on each step:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input string for the tool>

When you have the final answer, respond with:
Thought: <final reasoning>
Final Answer: <the answer>
"""

    scratchpad = ""
    for step in range(max_steps):
        if verbose:
            print(f"\n===== STEP {step + 1} =====")

        prompt = f"{system_prompt}\n\nQuestion: {query}\n{scratchpad}"
        response = llm(prompt)

        if verbose:
            print("LLM response:")
            print(response)

        final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
            if verbose:
                print(f"\n>>> FINAL ANSWER: {answer}")
            return answer

        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.DOTALL)

        if not action_match or not input_match:
            scratchpad += f"\n{response}\nObservation: Could not parse action. Use the required format."
            continue

        tool_name = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        if tool_name not in tools:
            observation = f"Error: unknown tool '{tool_name}'. Available: {list(tools)}"
        else:
            try:
                observation = str(tools[tool_name](tool_input))
            except Exception as e:
                observation = f"Error running {tool_name}: {e}"

        if verbose:
            print(f"Observation: {observation}")

        scratchpad += f"\n{response}\nObservation: {observation}\n"

    return None


# ---------- Tools ----------

def calculator(expr: str) -> str:
    """Evaluates a basic math expression like '2 + 2 * 3'."""
    return str(eval(expr, {"__builtins__": {}}, {}))


def search(q: str) -> str:
    """Looks up a fact about world capitals or science."""
    facts = {
        "capital of france": "Paris",
        "capital of japan": "Tokyo",
        "speed of light": "299,792,458 m/s",
        "population of paris": "2,165,000",
    }
    return facts.get(q.lower().strip(), "No result found.")


TOOLS = {"calculator": calculator, "search": search}


# ---------- Mock LLM (no API key needed) ----------

class MockLLM:
    """
    Pretends to be an LLM. Looks at the scratchpad and produces the next
    ReAct step. Good enough to demo the loop end-to-end.
    """

    def __call__(self, prompt: str) -> str:
        # Extract just the running scratchpad to decide what to do next
        question = re.search(r"Question:\s*(.+?)\n", prompt).group(1).strip()
        observations = re.findall(r"Observation:\s*(.+)", prompt)

        # Demo behavior: handle a multi-step question
        # "What is the capital of France, and what is 17 * 23?"
        if "capital of france" in question.lower() and "17 * 23" in question.lower():
            if len(observations) == 0:
                return (
                    "Thought: I need to find the capital of France first.\n"
                    "Action: search\n"
                    "Action Input: capital of France"
                )
            elif len(observations) == 1:
                return (
                    "Thought: Got Paris. Now I need to compute 17 * 23.\n"
                    "Action: calculator\n"
                    "Action Input: 17 * 23"
                )
            else:
                return (
                    f"Thought: I have both pieces. Capital is {observations[0]}, "
                    f"and 17 * 23 = {observations[1]}.\n"
                    f"Final Answer: The capital of France is {observations[0]} "
                    f"and 17 * 23 = {observations[1]}."
                )

        # Fallback: single calculator question
        if len(observations) == 0:
            # Try to extract a math expression from the question
            return (
                "Thought: This looks like a math problem.\n"
                "Action: calculator\n"
                f"Action Input: {question.replace('What is', '').replace('?', '').strip()}"
            )
        else:
            return f"Thought: Got the result.\nFinal Answer: {observations[-1]}"


# ---------- Real LLM (optional) ----------

def make_anthropic_llm():
    """
    Returns an llm(prompt) -> str function backed by Claude.
    Requires:  pip install anthropic
    And:       export ANTHROPIC_API_KEY=sk-...
    """
    from anthropic import Anthropic
    client = Anthropic()

    def llm(prompt: str) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return llm


# ---------- Main ----------

if __name__ == "__main__":
    # Pick which LLM to use
    USE_REAL_LLM = False  # flip to True after setting ANTHROPIC_API_KEY

    if USE_REAL_LLM:
        llm = make_anthropic_llm()
    else:
        llm = MockLLM()
        print("(Using MockLLM — it only knows two scripted queries.")
        print(" For arbitrary questions, set USE_REAL_LLM = True.)")
        print()
        print("Try one of:")
        print("  1. What is the capital of France, and what is 17 * 23?")
        print("  2. What is 144 / 12?")
        print()

    # Interactive loop — type your question, blank line to quit
    while True:
        try:
            query = input("Your question (blank to quit): ").strip()
        except EOFError:
            break
        if not query:
            break
        print("\n" + "#" * 60)
        print(f"QUERY: {query}")
        print("#" * 60)
        run_agent(query, TOOLS, llm)