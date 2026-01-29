"""Factory for creating scenarios."""

import argparse
from dataclasses import dataclass

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.audio.output_stream.base import AbstractAudioOutputStream
from conversa.scenarios.base import AbstractScenario
from conversa.scenarios.debugging_mirror import DebugMirrorScenario
from conversa.scenarios.file_narrator import FileNarratorScenario
from conversa.scenarios.talk import TalkScenario
from conversa.util.config import Config


@dataclass
class ScenarioDef:
    """Definition for a scenario."""

    scenario_class: type[AbstractScenario]


SCENARIOS: dict[str, ScenarioDef] = {
    "talk": ScenarioDef(TalkScenario),
    "debug_mirror": ScenarioDef(DebugMirrorScenario),
    "file_narrate": ScenarioDef(FileNarratorScenario),
}


def get_scenario(name: str) -> ScenarioDef:
    """Get a scenario definition by name.

    Args:
        name: Name of the scenario.

    Returns:
        ScenarioDef for the requested scenario.

    Raises:
        ValueError: If the scenario name is not found.
    """
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: '{name}'. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]


def create_scenario(
    name: str,
    input_stream: AbstractAudioInputStream,
    output_stream: AbstractAudioOutputStream,
    config: Config,
    args: argparse.Namespace,
) -> AbstractScenario:
    """Create and return a scenario instance.

    Args:
        name: Name of the scenario to create.
        input_stream: Audio input stream.
        output_stream: Audio output stream.
        config: Application configuration.
        args: Parsed command-line arguments.

    Returns:
        An initialized scenario instance ready to start.
    """
    scenario_def = get_scenario(name)
    return scenario_def.scenario_class(input_stream, output_stream, config, args)
