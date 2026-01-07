from dataclasses import dataclass, field

import gradio as gr
import numpy as np


@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    stopped: bool = False
    chunk_duration: float = 5.0  # Duration in seconds for each chunk
    conversation: list = field(default_factory=list)


def start_recording_user(state: AppState):
    if not state.stopped:
        return gr.Audio(recording=True)


def process_audio(audio: tuple, state: AppState):
    """Accumulate audio stream data and trigger playback when 3 seconds accumulated."""
    if audio is None:
        return None, state

    state.sampling_rate = audio[0]
    state.stream = (
        audio[1] if state.stream is None else np.concatenate((state.stream, audio[1]))
    )

    # Calculate required samples for chunk_duration seconds
    required_samples = int(state.sampling_rate * state.chunk_duration)

    print(f"Total samples: {len(state.stream)}, Required: {required_samples}")

    # Check if we have accumulated enough audio (3 seconds)
    if len(state.stream) >= required_samples:
        # Stop recording to play back
        print("Accumulated 3 seconds of audio, stopping recording for playback.")
        return gr.Audio(recording=False), state

    return None, state


def response(state: AppState):
    """Reverse and play back the recorded 3-second chunk, then restart recording."""
    if state.stream is None or len(state.stream) == 0:
        return

    # Extract exactly 3 seconds of audio
    chunk_samples = int(state.sampling_rate * state.chunk_duration)
    chunk = state.stream[:chunk_samples]

    # Reverse the audio chunk
    reversed_chunk = np.flip(chunk, axis=0)

    # Play the reversed chunk
    yield (state.sampling_rate, reversed_chunk), state

    # Remove the processed chunk from the stream buffer
    state.stream = (
        state.stream[chunk_samples:] if len(state.stream) > chunk_samples else None
    )

    # Continue recording unless stopped
    if not state.stopped:
        yield gr.Audio(recording=True), state


def add_new_words(state: AppState):
    """Add new words to the conversation history."""
    state.conversation.append(
        gr.ChatMessage(
            role="assistant", content="I am happy to provide you reversed audio."
        )
    )
    state.conversation.append(gr.ChatMessage(role="user", content="Thank you!"))
    return state, state.conversation


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

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.5,
        time_limit=None,  # No time limit for continuous recording
    )
    respond = input_audio.stop_recording(response, [state], [output_audio, state])
    respond.then(add_new_words, [state], [state, chatbot])

    restart = output_audio.stop(start_recording_user, [state], [input_audio])

    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(stopped=True), gr.Audio(recording=False)),
        None,
        [state, input_audio],
        cancels=[stream, respond],
    )

if __name__ == "__main__":
    demo.launch(share=True)
