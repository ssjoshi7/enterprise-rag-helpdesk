from rag_helpdesk import CLAUDE_API_KEY
import anthropic

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10,
    messages=[{"role": "user", "content": "Say hi"}]
)
print(msg.content[0].text)