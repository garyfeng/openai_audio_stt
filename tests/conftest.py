# pytest configuration and shared fixtures for mocking dify_plugin SDK
import sys
import types
import pytest

# Lightweight mock of dify_plugin so tests can run outside Dify
class _MockMsg:
    def __init__(self, kind: str, payload):
        self.type = kind
        if kind == "text":
            self.text = payload
        else:
            self.data = payload
    def to_dict(self):
        if self.type == "text":
            return {"type": "text", "text": getattr(self, "text", "")}
        return {"type": "json", "data": getattr(self, "data", {})}

class _MockTool:
    def __init__(self):
        self.runtime = types.SimpleNamespace(credentials={})
    def create_text_message(self, text: str):
        return _MockMsg("text", text)
    def create_json_message(self, data: dict):
        return _MockMsg("json", data)

# Install mocked modules before tests import tool code
mock_module = types.ModuleType("dify_plugin")
setattr(mock_module, "Tool", _MockTool)
entities_module = types.ModuleType("dify_plugin.entities")
entities_tool_module = types.ModuleType("dify_plugin.entities.tool")
setattr(entities_tool_module, "ToolInvokeMessage", _MockMsg)
sys.modules["dify_plugin"] = mock_module
sys.modules["dify_plugin.entities"] = entities_module
sys.modules["dify_plugin.entities.tool"] = entities_tool_module

@pytest.fixture
def make_tool():
    from tools.openai_audio import OpenaiAudioTool
    def _mk(creds: dict):
        t = OpenaiAudioTool()
        t.runtime = types.SimpleNamespace(credentials=creds)
        return t
    return _mk
