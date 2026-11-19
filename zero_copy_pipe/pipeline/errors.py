class PipelineError(RuntimeError):
    """Base error raised by the zero-copy pipeline."""


class PipelineClosedError(PipelineError):
    """Raised when accessing a pipeline or shared memory arena after shutdown/close."""


class PipelineWorkerError(PipelineError):
    """Raised when a spawned GPU or downstream CPU worker crashes or exits abnormally."""

    def __init__(self, worker_name: str, remote_traceback: str) -> None:
        self.worker_name = worker_name
        self.remote_traceback = remote_traceback
        super().__init__(f"{worker_name} failed:\n{remote_traceback}")
