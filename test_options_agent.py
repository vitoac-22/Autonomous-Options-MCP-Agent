"""
Unit tests for the OptionsExecutionAgent hybrid MCP-Featherless bridge.
"""
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from ai_agent.options_agent import OptionsExecutionAgent

class AsyncGeneratorMock:
    """Helper class to mock an async generator (stdio_client)."""
    def __init__(self, read_stream, write_stream):
        self.items = [(read_stream, write_stream)]
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            result = self.items[self.index]
            self.index += 1
            return result
        raise StopAsyncIteration

class AsyncMockResult:
    """Helper class to mock async MCP responses."""
    def __init__(self, tools=None, content=None):
        self.tools = tools or []
        self.content = content or []
        
    def __await__(self):
        yield self

class TestOptionsExecutionAgent(unittest.TestCase):

    @patch.dict('os.environ', {
        'FEATHERLESS_API_KEY': 'fake_key', 
        'ALPACA_API_KEY': 'fake', 
        'ALPACA_API_SECRET': 'fake'
    })
    @patch('ai_agent.options_agent.stdio_client')
    @patch('ai_agent.options_agent.ClientSession')
    @patch('ai_agent.options_agent.AsyncOpenAI')
    def test_agent_evaluates_and_dispatches(self, mock_openai, mock_client_session, mock_stdio):
        """Verifies the agent passes the payload to Featherless and processes the MCP tool call."""
        
        # Mocking the new async generator behavior of stdio_client
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_stdio.return_value = AsyncGeneratorMock(mock_read, mock_write)
        
        # Setup mock MCP Session
        mock_session_instance = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Mock MCP Tool List Response
        mock_tool = MagicMock()
        mock_tool.name = "dispatch_mleg_order"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {"payload_json": {"type": "string"}}}
        mock_session_instance.list_tools.return_value = AsyncMockResult([mock_tool])
        
        # Mock MCP Call Tool Response
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = '{"status": "success", "order_id": "1234"}'
        mock_session_instance.call_tool.return_value = AsyncMockResult(content=[mock_content])

        # Setup mock Featherless (OpenAI) Chat Response
        mock_llm_instance = AsyncMock()
        mock_openai.return_value = mock_llm_instance
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "dispatch_mleg_order"
        mock_tool_call.function.arguments = '{"payload_json": "{}"}'
        
        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tool_call]
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_llm_instance.chat.completions.create.return_value = mock_response

        # Execute
        agent = OptionsExecutionAgent()
        dummy_payload = {"order_class": "mleg"}
        result = agent.submit_mleg_payload(dummy_payload, rationale="Test rationale")

        # Assertions
        self.assertIn("success", result)
        mock_llm_instance.chat.completions.create.assert_called_once()
        mock_session_instance.call_tool.assert_called_once()

if __name__ == '__main__':
    unittest.main()