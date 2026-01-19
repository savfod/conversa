from dataclasses import dataclass, field
from threading import Thread
from typing import Any

import gradio as gr

from conversa.audio.stream_factory import create_input_stream, create_output_stream
from conversa.scenarios.talk import run_talk_scenario


@dataclass
class AppState:
    input_stream: Any = None
    output_stream: Any = None
    thread: Thread | None = None
    conversation: list = field(default_factory=list)


def start_scenario(state: AppState):
    """Initialize streams and start the scenario thread."""
    if state.thread and state.thread.is_alive():
        return state

    print("Initializing Gradio streams...")
    state.input_stream = create_input_stream("gradio", sample_rate=16000)
    state.output_stream = create_output_stream("gradio", sample_rate=16000)

    def thread_target():
        try:
            print("Starting scenario thread...")
            run_talk_scenario(state.input_stream, state.output_stream)
        except Exception as e:
            print(f"Scenario failed: {e}")
        finally:
            # We don't auto-close here to allow restarts, but standard scenario stops streams.
            pass

    state.thread = Thread(target=thread_target, daemon=True)
    state.thread.start()
    return state


def feed_input(audio: tuple, state: AppState):
    """Feed input audio from Gradio to the input stream."""
    if audio is None:
        return state

    sampling_rate, audio_data = audio
    if state.input_stream:
        state.input_stream.push_chunk(sampling_rate, audio_data)

    return state


def consume_output(state: AppState):
    """Yield output audio from the output stream to Gradio."""
    if not state.output_stream:
        return

    # Use the generator from the output stream adapter
    yield from state.output_stream.generate()


with gr.Blocks() as demo:
    state = gr.State(value=AppState())

    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Input Audio",
                sources="microphone",
                type="numpy",
                streaming=True,  # Enable streaming input
            )
        with gr.Column():
            chatbot = gr.Chatbot(
                label="Conversation"
            )  # Note: Chatbot update is tricky without separate event
            output_audio = gr.Audio(
                label="Output Audio",
                streaming=True,
                autoplay=True,
                interactive=False,  # Output only
            )

    start_btn = gr.Button("Start Conversation")

    # Start scenario on button click
    start_btn.click(start_scenario, [state], [state]).then(
        consume_output, [state], [output_audio]
    )

    # Feed input continuously
    input_audio.stream(feed_input, [input_audio, state], [state])

    # Note: Updating the chatbot would require another mechanism, e.g., gr.Timer
    # checking a shared history buffer. For now, we focus on audio unification.

if __name__ == "__main__":
    demo.queue().launch(share=True)
