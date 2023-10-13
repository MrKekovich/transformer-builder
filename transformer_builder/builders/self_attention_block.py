from typing import Optional, Callable


class SelfAttentionBlock:
    def __init__(
        self,
        before: Optional[Callable] = None,
        k: Optional[Callable] = None,
        q: Optional[Callable] = None,
        v: Optional[Callable] = None,
        after: Optional[Callable] = None,
        count: int = 1,
    ):
        pass
