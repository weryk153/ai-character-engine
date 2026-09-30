from ai_character_engine.llm.models import Message
from ai_character_engine.llm.provider import OpenAIResponsesClient


def test_openai_provider_maps_event_at_boundary() -> None:
    items = OpenAIResponsesClient._to_response_input(
        [Message(role="event", content="type: timer\ncontent: break time")]
    )

    assert items == [
        {
            "role": "user",
            "content": "[Environment event]\ntype: timer\ncontent: break time",
        }
    ]
