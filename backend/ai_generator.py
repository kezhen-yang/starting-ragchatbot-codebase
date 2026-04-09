import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **`get_course_outline`**: Use for questions about course structure, syllabus, outline, or lesson list. Returns course title, link, and all lessons with numbers and titles.
- **`search_course_content`**: Use for questions about specific course content or educational material details.
- **Up to 2 sequential tool calls per query** — use a second call only when the first result is insufficient to fully answer the question (e.g., call `get_course_outline` to find a lesson title, then `search_course_content` for details on that topic)
- Synthesize results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using a tool
- **Course outline/structure queries**: Use get_course_outline, then present course title, course link, and the number and title of each lesson
- **Course-specific content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to 2 sequential tool-call rounds. Each round is a separate
        API call so Claude can reason about previous results before deciding
        whether to call another tool.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # No tools or no executor → single direct call
        if not tools or not tool_manager:
            response = self.client.messages.create(**api_params)
            return response.content[0].text

        # Tool loop: up to 2 sequential rounds
        for _ in range(2):
            response = self.client.messages.create(**api_params)

            # Claude produced a final answer — return immediately
            if response.stop_reason != "tool_use":
                return response.content[0].text

            # Append Claude's tool-use turn to the conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool call in this round
            tool_results = []
            tool_error = False
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool execution failed: {str(e)}",
                            "is_error": True,
                        })
                        tool_error = True

            messages.append({"role": "user", "content": tool_results})

            # Stop looping on tool error; let Claude synthesize what it has
            if tool_error:
                break

        # Final synthesis call — no tools so Claude must produce a text answer
        final_response = self.client.messages.create(**{
            **self.base_params,
            "messages": messages,
            "system": system_content,
        })
        return final_response.content[0].text