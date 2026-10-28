from typing import Any, Callable

def checkpoint(
    function: Callable[..., Any],
    *args: Any,
    use_reentrant: bool = ...,
    **kwargs: Any,
) -> Any: ...
