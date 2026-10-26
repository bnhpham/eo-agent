"""
Command-line smoke test of the agent: python -m eo_agent.cli "What disasters are happening in South America?"
"""

import argparse
import sys
import uuid

from eo_agent.agent.runtime import AgentRuntime
from eo_agent.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the disaster-monitoring agent one or more questions (same session).")
    parser.add_argument("questions", nargs="+", help="Questions, asked in order within one conversation")
    args = parser.parse_args()
    # Answers contain characters such as the minus sign that the Windows console code page cannot print.
    sys.stdout.reconfigure(encoding="utf-8")

    runtime = AgentRuntime(get_settings())
    runtime.start()
    try:
        if runtime.startup_error:
            print(f"Startup problem: {runtime.startup_error}", file=sys.stderr)
            return 1
        session_id = uuid.uuid4().hex
        for question in args.questions:
            print(f"\n>>> {question}")
            result = runtime.run_turn(session_id, question)
            print(result.answer)
            for note in result.notes:
                print(f"[note] {note}")
            print(f"[tools] {', '.join(result.artifacts.tool_calls) or 'none'}")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
