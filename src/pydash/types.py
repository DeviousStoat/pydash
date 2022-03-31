from typing import Any, Callable, Mapping, ParamSpec, Protocol, Iterable, TypeVar


T = TypeVar("T")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

P = ParamSpec("P")


ArrayCallableT = Callable[[T], T2] | Callable[[T, int], T2] | Callable[[T, int, list[T]], T2]
IterateeObjT = int | str | list | tuple | dict | None
# TODO: can use `Sequence`?
IterateeT = ArrayCallableT[T, T2] | int | str | list | tuple | dict
CollectionT = TypeVar("CollectionT", Mapping, Iterable)


# WARN: order matters for good generic resolution
DictIterateeT = Callable[[T2, T, dict[T, T2]], T3] | Callable[[T2, T], T3] | Callable[[T2], T3]


# WARN: order matters for good generic resolution
# ListIterateeT = Callable[[T, int, Iterable[T]], T2] | Callable[[T, int], T2] | Callable[[T], T2]
ListIterateeT = Callable[[T, int, list[T]], T2] | Callable[[T, int], T2] | Callable[[T], T2]


class Representable(Protocol):
    def __repr__(self) -> str:
        ...


Number = int | float
NumberT = TypeVar("NumberT", bound=Number)
WithT = TypeVar("WithT", contravariant=True)
ToT = TypeVar("ToT", covariant=True)


class Addable(Protocol[WithT, ToT]):
    def __add__(self, x: WithT, /) -> ToT:
        ...


class SupportsDunderLT(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


class SupportsDunderGT(Protocol):
    def __gt__(self, __other: Any) -> bool:
        ...


SupportsRichComparison = SupportsDunderLT | SupportsDunderGT
SupportsRichComparisonT = TypeVar("SupportsRichComparisonT", bound=SupportsRichComparison)
