"""Stateless Responses API bridge for the Firewall engine's local tool history."""

from openai.types.chat import ChatCompletion

from cloudops_hub.config import Settings
from cloudops_hub.firewall_model_data import active_references
from cloudops_hub.model_usage import complete, restore_response
from cloudops_hub.retrieval import completion_parameters


class FirewallResponses:
    def __init__(self):
        # Keep encrypted reasoning only within this tool loop; never enable provider storage.
        self.reasoning: dict[str, list[dict]] = {}

    def request(
        self, client, settings: Settings, messages: list[dict], tools: list[dict], *,
        require_tool: bool = False,
    ):
        references = active_references()
        if references is not None:
            messages = references.messages(messages)
        inputs: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": message["content"],
                    }
                )
                continue
            calls = message.get("tool_calls") or []
            if calls:
                inputs.extend(self.reasoning.get(calls[0]["id"], []))
            if message.get("content"):
                inputs.append({"role": role, "content": message["content"]})
            for call in calls:
                inputs.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        parameters = completion_parameters(settings, settings.troubleshooting_output_tokens)
        if "reasoning_effort" in parameters:
            parameters["reasoning"] = {"effort": parameters.pop("reasoning_effort")}
        if "max_completion_tokens" in parameters:
            parameters["max_output_tokens"] = parameters.pop("max_completion_tokens")
        response = complete(
            client,
            settings,
            "firewall_live",
            responses_api=True,
            model=settings.azure_openai_deployment,
            input=inputs,
            tools=[{"type": "function", **tool["function"], "strict": False} for tool in tools],
            tool_choice="required" if tools and require_tool else "auto" if tools else "none",
            parallel_tool_calls=False,
            include=["reasoning.encrypted_content"],
            **parameters,
        )
        if response.status != "completed":
            raise RuntimeError("Firewall analysis was incomplete; no partial answer was returned")
        reasoning = [
            item.model_dump(exclude_none=True)
            for item in response.output
            if item.type == "reasoning"
        ]
        calls = []
        for item in response.output:
            if item.type == "function_call":
                calls.append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": item.arguments},
                    }
                )
        if calls:
            self.reasoning[calls[0]["id"]] = reasoning
        result = ChatCompletion(
            id=response.id,
            created=int(response.created_at),
            model=response.model,
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": response.output_text,
                        "tool_calls": calls or None,
                    },
                }
            ],
        )
        if references is not None and references.restore_outputs:
            restore_response(references, result)
        return result
