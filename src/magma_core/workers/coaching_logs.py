"""Run-owned trace storage, shared by local and remote coaching calls."""
from pathlib import Path
from threading import Lock

from magma_core.protocol.agent_coaching import CoachingLog


class CoachingLogWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.lock = Lock()
        self.counter = 0

    def __call__(self, log: CoachingLog) -> Path:
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            while True:
                self.counter += 1
                path = self.directory / f"{self.counter:06d}_{log.coaching_type}.md"
                try:
                    with path.open("x", encoding="utf-8") as output:
                        output.write(log.content)
                    return path
                except FileExistsError:
                    continue
