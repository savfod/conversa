from dataclasses import dataclass, field

import gradio as gr

from conversa.features.llm_api import call_llm
from conversa.generated.speech_api import speech_to_text, text_to_speech


@dataclass
class AppState:
    # it can be useful for some settings
    language: str = "en"
    conversation: list = field(default_factory=list)


def response(audio_tuple: tuple, state: AppState):
    """Play back recorded audio in 1-second chunks with 1-second delay."""
    sampling_rate, audio = audio_tuple
    if len(audio) > sampling_rate * 60:
        audio = audio[-sampling_rate * 60 :]  # Keep only the last 60 seconds
        state.conversation.append(
            {"role": "assistant", "content": "[Truncated audio to last 60 seconds]"}
        )

    text = speech_to_text(audio, sample_rate=sampling_rate, language=state.language)
    state.conversation.append({"role": "user", "content": text})

    answer = call_llm(
        text, sys_prompt="You are a helpful assistant.", history=state.conversation
    )
    state.conversation.append({"role": "assistant", "content": answer})

    output_audio = (
        16000,
        text_to_speech(answer, instructions="Speak clearly and slowly."),
    )
    return output_audio, state, state.conversation


def clear_input_stream():
    print("Clearing input stream")
    return None


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Input Audio", sources="microphone", type="numpy"
            )
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation")
            output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)

    state = gr.State(value=AppState())

    respond = input_audio.stop_recording(
        response, [input_audio, state], [output_audio, state, chatbot]
    )
    respond.then(clear_input_stream, [], [input_audio])

if __name__ == "__main__":
    demo.launch(share=True)
