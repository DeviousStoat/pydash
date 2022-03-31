from typing import Callable
from abc import ABC, abstractmethod

import pydash as pyd


class AllFuncs(ABC):
    @abstractmethod
    def _wrap(self, func) -> Callable:
        ...

    def __getattr__(self, name: str) -> Callable:
        method = getattr(pyd, name, None)

        if not callable(method) and not name.endswith("_"):
            # Alias method names not ending in underscore to their underscore
            # counterpart. This allows chaining of functions like "map_()"
            # using "map()" instead.
            method = getattr(pyd, name + "_", None)

        if not callable(method):
            raise pyd.InvalidMethod(f"Invalid pydash method: {name}")

        return self._wrap(method)
