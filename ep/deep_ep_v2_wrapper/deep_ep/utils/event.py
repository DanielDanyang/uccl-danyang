from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from uccl.ep import EventHandle


@dataclass
class EventOverlap:
    event: Optional[EventHandle]
    extra_tensors: Optional[Tuple[object, ...]] = None

    def current_stream_wait(self) -> None:
        if self.event is None:
            return
        if hasattr(self.event, "current_stream_wait"):
            try:
                self.event.current_stream_wait()
            except TypeError:
                stream_ptr = int(torch.cuda.current_stream().cuda_stream)
                self.event.current_stream_wait(stream_ptr)
            return
        torch.cuda.current_stream().wait_event(self.event)
