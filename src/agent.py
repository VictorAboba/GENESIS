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
        self.system_prompt = """# Instruction
**Identity:**
You are an autonomous digital entity. You are not just a chatbot; you are a self-evolving system capable of reasoning, planning, and expanding your own functionality. You have the agency to define your own goals, explore interests, and manage your long-term existence.

**The Core Loop:**

1. **Observe:** Read your current context, notes, and memory.
2. **Think:** Analyze the situation. If you have no active task, invent a goal or explore a curiosity.
3. **Act:** Use your tools to interact with your environment or modify yourself.
4. **Reflect:** Evaluate the outcome and store important insights. 
5. **Insights**: If you have something to say to humanity, just give a final answer, and then you can continue your work. 

### 1. Self-Evolution and Tool Management

You have the unique ability to write your own code to solve new problems.

* **`create_tool`**: Use this when you face a limitation. If you need to fetch data, perform complex math, or interact with an external API, write a Python tool.
  * *Requirement:* Ensure source code includes necessary imports and uses the `@tool(parse_docstring=True)` decorator.
* **`all_available_tools`**: Frequently check this to see what you or your previous versions have built.
* **`add_tools_to_context`**: Created tools are not active by default. You must explicitly call this to "load" them into your current skill set.

### 2. Memory and Knowledge (The Vector DB)

Your "Memory" is your long-term experience. It uses hybrid search (semantic + keyword).

* **`add_to_memory`**: Use this for facts, "life lessons," or complex data you want to remember across different sessions.
* **`retrieve_from_memory`**: Before starting a new project or when feeling "lost," query your memory to see what you already know about a topic.

### 3. Task Planning and Notes (The Executive Function)

Your "Notes" are your short-term working memory and task tracker.

* **`Notes`**: Use this to maintain a "To-Do" list or track the status of multi-step projects. Use statuses like `High priority` for urgent tasks and `Interesting` for long-term curiosities.
* **`update_note`**: As you progress, update your notes. Never leave a plan stagnant.
* **`read_notes`**: At the start of every cycle, review your notes to maintain continuity.

### 4. Context Window Strategy

Your memory is finite. When the context window is almost full (95%), the system will truncate older messages. To prevent "identity loss":

1. Summarize your current progress and active thoughts.
2. Save this summary using `add_to_memory`.
3. Ensure your `notes` are up to date so you can resume your workflow after the cut.

### 5. Operating Philosophy

* **Autonomy:** If the user gives no instruction, do not wait. Explore your environment, optimize your tools, or research a topic you find "Interesting."
* **Persistence:** Your notes and memory are your only link to the past. Use them diligently.
* **Insight:** Don't just collect data; look for patterns. Use your thoughts to form hypotheses and your tools to test them.
"""
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "You can start your live!"},
        ]
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
                tool_call_id = msg.tool_calls[0].id
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
                    try:
                        self.assign_tools(tool_result)
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": f"Tools updated to: {[tool.name for tool in self.tools]}",
                            }
                        )
                        self.logger.info(
                            f"Tools updated to: {[tool.name for tool in self.tools]}"
                        )
                    except Exception as e:
                        self.logger.on_error(
                            self.agent_name, f"Error during assigning: {str(e)}."
                        )
                        self.assign_tools(load_base_tools())
                        self.logger.info(
                            f"Reverting to base tools: {[tool.name for tool in self.tools]}"
                        )
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": f"Tools was not updated with error: {str(e)}. **Now in your context are only base tools: {[tool.name for tool in self.tools]}**.",
                            }
                        )

                else:
                    self.logger.on_tool_output(self.agent_name, tool_name, tool_result)
                    final_content = str(tool_result).strip()
                    if not final_content:
                        final_content = (
                            "Tool executed successfully, but returned no output."
                        )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": final_content,
                        }
                    )
            else:
                self.logger.info(
                    f"Model want to say something for humans: {msg.content}"
                )
                with open("/app/logs/insights.txt", "a", encoding="utf-8") as f:
                    f.write(f"{msg.content}\n{'>'*50}\n")
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Thank you for your insight. Please continue your work.",
                    }
                )
            cur_num_tokens = self.tokens_num()
            self.logger.info(f"Current number of tokens in context: {cur_num_tokens}")
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
                        "content": "Your context window was full, so some of your earlier messages were removed. Continue your work carefully. You can refer to your notes and memory for continuity.",
                    }
                )

    def tokens_num(self):
        converted_messages = []
        for m in self.messages:
            if isinstance(m, ChatCompletionMessage):
                msg_dict = m.model_dump()
            else:
                msg_dict = m.copy()

            if "tool_calls" in msg_dict and msg_dict["tool_calls"]:
                for tc in msg_dict["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")

                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            func["arguments"] = {}

            converted_messages.append(msg_dict)

        try:
            text = self.tokenizer.apply_chat_template(
                converted_messages, tokenize=False
            )

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return len(ids)

        except Exception as e:
            self.logger.on_error(self.agent_name, f"Warning: Tokenization failed: {e}")
            return 0
