import json
import datetime
from textwrap import shorten


class BeautifulLogger:
    class Colors:
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        GREY = "\033[90m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"
        RESET = "\033[0m"

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.st_status = None

    def _timestamp(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _print_box(self, title, content, color):
        print(f"{color}╭{'─' * 50}")
        print(f"│ {self.Colors.BOLD}{title}{self.Colors.RESET}{color}")
        print(f"│ {self._timestamp()}")
        print(f"├{'─' * 50}")
        for line in content.split("\n"):
            print(f"│ {line}")
        print(f"╰{'─' * 50}{self.Colors.RESET}")

    def on_agent_start(self, agent_name: str, prompt: str):
        print(
            f"\n{self.Colors.HEADER}{self.Colors.BOLD}🤖 AGENT START: {agent_name.upper()}{self.Colors.RESET}"
        )
        print(
            f"{self.Colors.GREY}Prompt: {shorten(prompt, width=100, placeholder='...')}{self.Colors.RESET}"
        )
        print(f"{self.Colors.HEADER}{'='*60}{self.Colors.RESET}")

    def on_thought(self, agent_name: str, thought: str):
        if thought:
            print(f"{self.Colors.CYAN}💭 [{agent_name}] Thinking:{self.Colors.RESET}")
            print(f"{self.Colors.CYAN}{thought.strip()}{self.Colors.RESET}\n")

    def on_tool_call(self, agent_name: str, tool_name: str, args: dict):
        args_str = json.dumps(args, ensure_ascii=False, indent=2)
        print(
            f"{self.Colors.YELLOW}🛠️  [{agent_name}] Calling Tool: {self.Colors.BOLD}{tool_name}{self.Colors.RESET}"
        )
        formatted_args = "\n".join([f"    {line}" for line in args_str.split("\n")])
        print(f"{self.Colors.YELLOW}{formatted_args}{self.Colors.RESET}")

    def on_tool_output(self, agent_name: str, tool_name: str, result: str):
        preview = shorten(
            str(result),
            width=300,
            placeholder=f"... [truncated {len(str(result))} chars]",
        )
        print(
            f"{self.Colors.GREEN}✅ [{agent_name}] Tool Output ({tool_name}):{self.Colors.RESET}"
        )
        print(f"{self.Colors.GREY}{preview}{self.Colors.RESET}\n")

    def on_agent_response(self, agent_name: str, response: str):
        self._print_box(f"FINAL ANSWER ({agent_name})", response, self.Colors.BLUE)

    def on_error(self, agent_name: str, error: str):
        print(f"\n{self.Colors.RED}❌ [{agent_name}] ERROR: {error}{self.Colors.RESET}")

    def info(self, message: str):
        print(f"{self.Colors.GREY}ℹ️  {message}{self.Colors.RESET}")
