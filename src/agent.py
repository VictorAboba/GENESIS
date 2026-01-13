import json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from transformers import AutoTokenizer

from tools import load_base_tools, basetools_to_jsons
from logger import BeautifulLogger


class AutonomousAgent:
    def __init__(
        self, client: OpenAI, model: str, hf_model: str, max_context_tokens: int
    ) -> None:
        self.client = client
        self.model = model
        self.tools = load_base_tools()
        self.json_tools = basetools_to_jsons(self.tools)
        self.name_to_tool = {tool.name: tool for tool in self.tools}
        self.system_prompt = """"""
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.logger = BeautifulLogger(verbose=True)
        self.agent_name = "autonomous_agent"
        self.max_context_tokens = max_context_tokens

    def assign_tools(self, new_tools):
        self.tools = new_tools
        self.json_tools = basetools_to_jsons(new_tools)
        self.name_to_tool = {tool.name: tool for tool in new_tools}

    def live(self):
        while True:
            msg = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.json_tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
                .choices[0]
                .message
            )
            self.messages.append(msg)

            if msg.tool_calls:
                self.logger.on_thought(self.agent_name, msg.content)
                tool_call = msg.tool_calls[0].function
                tool_name = tool_call.name
                try:
                    tool_args = json.loads(tool_call.arguments)
                    self.logger.on_tool_call(self.agent_name, tool_name, tool_args)
                    tool_result = self.name_to_tool[tool_name].invoke(tool_args)
                except Exception as e:
                    self.logger.on_error(self.agent_name, str(e))
                    tool_result = str(e)
                if tool_name == "add_tools_to_context":
                    self.assign_tools(tool_result)
                    self.messages.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Tools updated to: {[tool.name for tool in self.tools]}",
                        }
                    )
                    self.logger.info(
                        f"Tools updated to: {[tool.name for tool in self.tools]}"
                    )
                else:
                    self.logger.on_tool_output(self.agent_name, tool_name, tool_result)
                    self.messages.append(
                        {"role": "tool", "name": tool_name, "content": tool_result}
                    )
            cur_num_tokens = self.tokens_num()
            if cur_num_tokens > 0.8 * self.max_context_tokens:
                self.messages.append(
                    {
                        "role": "system",
                        "content": "Your context window is almost full. It will be cut soon, so be careful!",
                    }
                )
            elif cur_num_tokens > 0.95 * self.max_context_tokens:
                self.logger.info("Context window exceeded limit. Cutting messages.")
                to_save = int(len(self.messages) / 2)
                self.messages = [self.messages[0]] + self.messages[to_save:]
                self.logger.info(f"Messages cut. {len(self.messages)} messages remain.")
                self.messages.append(
                    {
                        "role": "system",
                        "content": "Your context window was full, so some of your earlier messages were removed. Continue your work carefully.",
                    }
                )

    def tokens_num(self):
        converted_messages = [
            m.model_dump() if isinstance(m, ChatCompletionMessage) else m
            for m in self.messages
        ]
        text = self.tokenizer.apply_chat_template(converted_messages, tokenize=False)
        inputs = self.tokenizer.tokenize([text])
        return len(inputs["input_ids"][0]) if inputs["input_ids"] else 0
