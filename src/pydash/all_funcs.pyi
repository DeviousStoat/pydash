
'''Generated from the `build_class` script'''
# pyright: reportWildcardImportFromLibrary=false

import datetime
import re
from abc import ABC, abstractmethod
from typing import *

import pydash as pyd

from pydash.helpers import Unset, UNSET
from pydash.functions import After, Ary, Before, Once, Spread, Throttle
from pydash.predicates import RegExp

from pydash.chaining import Chain

ValueT = TypeVar("ValueT", covariant=True)

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


class AllFuncs(ABC):
    def chunk(self: 'Chain[list[T]]', size: int=1) -> 'Chain[list[list[T]]]':
        return self._wrap(pyd.chunk)(size)

    def compact(self: 'Chain[list[T]]') -> 'Chain[list[T]]':
        return self._wrap(pyd.compact)()

    def difference(self: 'Chain[list[T]]', *others: Sequence) -> 'Chain[list[T]]':
        return self._wrap(pyd.difference)(*others)

    def difference_by(self: 'Chain[list[T]]', *others: Sequence, iteratee: IterateeT[T, Any] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.difference_by)(*others, iteratee=iteratee)

    def difference_with(self: 'Chain[list[T]]', *others: Sequence[T2], comparator: Callable[[T, T2], bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.difference_with)(*others, comparator=comparator)

    def drop(self: 'Chain[list[T]]', n: int=1) -> 'Chain[list[T]]':
        return self._wrap(pyd.drop)(n)

    def drop_right(self: 'Chain[list[T]]', n: int=1) -> 'Chain[list[T]]':
        return self._wrap(pyd.drop_right)(n)

    def drop_right_while(self: 'Chain[list[T]]', predicate: ArrayCallableT[T, bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.drop_right_while)(predicate)

    def drop_while(self: 'Chain[list[T]]', predicate: ArrayCallableT[T, bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.drop_while)(predicate)

    def duplicates(self: 'Chain[list[T]]', iteratee: IterateeT[T, T] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.duplicates)(iteratee)

    def fill(self: 'Chain[list[T]]', value: T, start: int=0, end: int | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.fill)(value, start, end)

    def find_index(self: 'Chain[list[T]]', predicate: ArrayCallableT[T, bool] | None=None) -> 'Chain[int]':
        return self._wrap(pyd.find_index)(predicate)

    def find_last_index(self: 'Chain[list[T]]', predicate: ArrayCallableT[T, bool] | None=None) -> 'Chain[int]':
        return self._wrap(pyd.find_last_index)(predicate)

    def flatten(self: 'Chain[Iterable[Iterable[T] | T]]') -> 'Chain[list[T]]':
        return self._wrap(pyd.flatten)()

    def flatten_deep(self: 'Chain[Iterable]') -> 'Chain[list]':
        return self._wrap(pyd.flatten_deep)()

    def flatten_depth(self: 'Chain[Iterable]', depth: int=1) -> 'Chain[list]':
        return self._wrap(pyd.flatten_depth)(depth)

    def from_pairs(self: 'Chain[Iterable[tuple[T, T2] | list[T | T2]]]') -> 'Chain[dict[T, T2]]':
        return self._wrap(pyd.from_pairs)()

    def head(self: 'Chain[Sequence[T]]') -> 'Chain[T | None]':
        return self._wrap(pyd.head)()

    def index_of(self: 'Chain[list[T]]', value: T, from_index: int=0) -> 'Chain[int]':
        return self._wrap(pyd.index_of)(value, from_index)

    def initial(self: 'Chain[list[T]]') -> 'Chain[list[T]]':
        return self._wrap(pyd.initial)()

    def intercalate(self: 'Chain[Iterable[Iterable[T] | T]]', separator: T2 | Iterable[T2]) -> 'Chain[list[T | T2]]':
        return self._wrap(pyd.intercalate)(separator)

    def intersection(self: 'Chain[list[T]]', *others: list[T]) -> 'Chain[list[T]]':
        return self._wrap(pyd.intersection)(*others)

    @overload
    def intersection_by(self: 'Chain[list[T]]', *others: tuple[()], iteratee: ArrayCallableT[T, T2] | None=None) -> 'Chain[list[T]]':
        ...

    @overload
    def intersection_by(self: 'Chain[list[T]]', *others: list[T], iteratee: ArrayCallableT[T, T2] | None=None) -> 'Chain[list[T | T2]]':
        ...

    def intersection_by(self, *others, iteratee=None):
        return self._wrap(pyd.intersection_by)(*others, iteratee=iteratee)

    @overload
    def intersection_with(self: 'Chain[list[T]]', *others: tuple[()], comparator: ArrayCallableT[T, bool] | None=None) -> 'Chain[list[T]]':
        ...

    @overload
    def intersection_with(self: 'Chain[list[T]]', *others: list[T], comparator: ArrayCallableT[T, bool] | None=None) -> 'Chain[list[T]]':
        ...

    def intersection_with(self, *others, comparator=None):
        return self._wrap(pyd.intersection_with)(*others, comparator=comparator)

    def intersperse(self: 'Chain[Iterable[T]]', separator: T2) -> 'Chain[list[T | T2]]':
        return self._wrap(pyd.intersperse)(separator)

    def last(self: 'Chain[Sequence[T]]') -> 'Chain[T | None]':
        return self._wrap(pyd.last)()

    def last_index_of(self: 'Chain[Sequence[T]]', value: T, from_index: SupportsInt | None=None) -> 'Chain[int]':
        return self._wrap(pyd.last_index_of)(value, from_index)

    def mapcat(self: 'Chain[list[T]]', iteratee: IterateeT[T, list[T2]] | None=None) -> 'Chain[list[T2]]':
        return self._wrap(pyd.mapcat)(iteratee)

    def nth(self: 'Chain[Sequence[T]]', pos: int=0) -> 'Chain[T | None]':
        return self._wrap(pyd.nth)(pos)

    def pop(self: 'Chain[list[T]]', index: int=-1) -> 'Chain[T]':
        return self._wrap(pyd.pop)(index)

    def pull(self: 'Chain[list[T]]', *values: Any) -> 'Chain[list[T]]':
        return self._wrap(pyd.pull)(*values)

    def pull_all(self: 'Chain[list[T]]', values: Sequence) -> 'Chain[list[T]]':
        return self._wrap(pyd.pull_all)(values)

    def pull_all_by(self: 'Chain[list[T]]', values: Sequence, iteratee: IterateeT[T, T2] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.pull_all_by)(values, iteratee)

    def pull_all_with(self: 'Chain[list[T]]', values: Sequence[T2], comparator: Callable[[T, T2], bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.pull_all_with)(values, comparator)

    def pull_at(self: 'Chain[list[T]]', *indexes: int) -> 'Chain[list[T]]':
        return self._wrap(pyd.pull_at)(*indexes)

    def push(self: 'Chain[list[T]]', *items: T) -> 'Chain[list[T]]':
        return self._wrap(pyd.push)(*items)

    def remove(self: 'Chain[list[T]]', predicate: ArrayCallableT[T, bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.remove)(predicate)

    def reverse(self: 'Chain[Sequence[T]]') -> 'Chain[Sequence[T]]':
        return self._wrap(pyd.reverse)()

    def shift(self: 'Chain[list[T]]') -> 'Chain[T]':
        return self._wrap(pyd.shift)()

    def slice_(self: 'Chain[list[T]]', start: int=0, end: int | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.slice_)(start, end)

    slice = slice_

    @overload
    def sort(self: 'Chain[list[T]]', comparator: Callable[[T, T], int], key: None=None, reverse: bool=False) -> 'Chain[list[T]]':
        ...

    @overload
    def sort(self: 'Chain[list[T]]', comparator: None=None, key: Callable[[T], Any] | None=None, reverse: bool=False) -> 'Chain[list[T]]':
        ...

    def sort(self, comparator=None, key=None, reverse=False):
        return self._wrap(pyd.sort)(comparator, key, reverse)

    def sorted_index(self: 'Chain[list[SupportsRichComparisonT]]', value: SupportsRichComparisonT) -> 'Chain[int]':
        return self._wrap(pyd.sorted_index)(value)

    def sorted_index_by(self: 'Chain[list[SupportsRichComparisonT]]', value: SupportsRichComparisonT, iteratee: IterateeT[SupportsRichComparisonT, Any] | None=None) -> 'Chain[int]':
        return self._wrap(pyd.sorted_index_by)(value, iteratee)

    def sorted_index_of(self: 'Chain[list[SupportsRichComparisonT]]', value: SupportsRichComparisonT) -> 'Chain[int]':
        return self._wrap(pyd.sorted_index_of)(value)

    def sorted_last_index(self: 'Chain[list[SupportsRichComparisonT]]', value: SupportsRichComparisonT) -> 'Chain[int]':
        return self._wrap(pyd.sorted_last_index)(value)

    def sorted_last_index_by(self: 'Chain[list[SupportsRichComparisonT]]', value: SupportsRichComparisonT, iteratee: IterateeT[SupportsRichComparisonT, Any] | None=None) -> 'Chain[int]':
        return self._wrap(pyd.sorted_last_index_by)(value, iteratee)

    def sorted_last_index_of(self: 'Chain[list[SupportsRichComparisonT]]', value: SupportsRichComparisonT) -> 'Chain[int]':
        return self._wrap(pyd.sorted_last_index_of)(value)

    def sorted_uniq(self: 'Chain[list[SupportsRichComparisonT]]') -> 'Chain[list[SupportsRichComparisonT]]':
        return self._wrap(pyd.sorted_uniq)()

    def sorted_uniq_by(self: 'Chain[list[SupportsRichComparisonT]]', iteratee: IterateeT[SupportsRichComparisonT, Any] | None=None) -> 'Chain[list[SupportsRichComparisonT]]':
        return self._wrap(pyd.sorted_uniq_by)(iteratee)

    @overload
    def splice(self: 'Chain[str]', start: int, count: int | None=None, *items: str) -> 'Chain[str]':
        ...

    @overload
    def splice(self: 'Chain[list[T]]', start: int, count: int | None=None, *items: T) -> 'Chain[list[T]]':
        ...

    def splice(self, start, count=None, *items):
        return self._wrap(pyd.splice)(start, count, *items)

    def split_at(self: 'Chain[list[T]]', index: int) -> 'Chain[list[list[T]]]':
        return self._wrap(pyd.split_at)(index)

    def tail(self: 'Chain[list[T]]') -> 'Chain[list[T]]':
        return self._wrap(pyd.tail)()

    def take(self: 'Chain[list[T]]', n: int=1) -> 'Chain[list[T]]':
        return self._wrap(pyd.take)(n)

    def take_right(self: 'Chain[list[T]]', n: int=1) -> 'Chain[list[T]]':
        return self._wrap(pyd.take_right)(n)

    def take_right_while(self: 'Chain[list[T]]', predicate: IterateeT[T, bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.take_right_while)(predicate)

    def take_while(self: 'Chain[list[T]]', predicate: IterateeT[T, bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.take_while)(predicate)

    def union(self: 'Chain[list[T]]', *others: list[T]) -> 'Chain[list[T]]':
        return self._wrap(pyd.union)(*others)

    def union_by(self: 'Chain[list[T]]', *others: list[T], iteratee: IterateeT[T, Any]) -> 'Chain[list[T]]':
        return self._wrap(pyd.union_by)(*others, iteratee=iteratee)

    def union_with(self: 'Chain[list[T]]', *others: list[T], comparator: Callable[[T, T], bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.union_with)(*others, comparator=comparator)

    def uniq(self: 'Chain[Iterable[T]]') -> 'Chain[list[T]]':
        return self._wrap(pyd.uniq)()

    def uniq_by(self: 'Chain[Iterable[T]]', iteratee: IterateeT[T, Any] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.uniq_by)(iteratee)

    def uniq_with(self: 'Chain[list[T]]', comparator: Callable[[T, T], bool] | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.uniq_with)(comparator)

    def unshift(self: 'Chain[list[T]]', *items: T) -> 'Chain[list[T]]':
        return self._wrap(pyd.unshift)(*items)

    def unzip(self: 'Chain[Iterable[Iterable[T]]]') -> 'Chain[list[list[T]]]':
        return self._wrap(pyd.unzip)()

    def unzip_with(self, iteratee):
        return self._wrap(pyd.unzip_with)(iteratee)

    def without(self: 'Chain[Iterable[T]]', *values: Any) -> 'Chain[list[T]]':
        return self._wrap(pyd.without)(*values)

    def xor(self: 'Chain[list[T]]', *lists: list[T]) -> 'Chain[list[T]]':
        return self._wrap(pyd.xor)(*lists)

    def xor_by(self, *lists, **kwargs):
        return self._wrap(pyd.xor_by)(*lists, **kwargs)

    def xor_with(self, *lists, **kwargs):
        return self._wrap(pyd.xor_with)(*lists, **kwargs)

    def zip_object(self, values=None):
        return self._wrap(pyd.zip_object)(values)

    def zip_object_deep(self, values=None):
        return self._wrap(pyd.zip_object_deep)(values)

    def tap(self, interceptor):
        return self._wrap(pyd.tap)(interceptor)

    def thru(self, interceptor):
        return self._wrap(pyd.thru)(interceptor)

    @overload
    def at(self: 'Chain[Mapping[T, T2]]', *paths: T | Iterable[T]) -> 'Chain[list[T2]]':
        ...

    @overload
    def at(self: 'Chain[Iterable[T]]', *paths: int | Iterable[int]) -> 'Chain[list[T]]':
        ...

    def at(self, *paths):
        return self._wrap(pyd.at)(*paths)

    @overload
    def count_by(self: 'Chain[Mapping[Any, T2]]', iteratee: None=None) -> 'Chain[dict[T2, int]]':
        ...

    @overload
    def count_by(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, T3]) -> 'Chain[dict[T3, int]]':
        ...

    @overload
    def count_by(self: 'Chain[Iterable[T]]', iteratee: None=None) -> 'Chain[dict[T, int]]':
        ...

    @overload
    def count_by(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, T2]) -> 'Chain[dict[T2, int]]':
        ...

    def count_by(self, iteratee=None):
        return self._wrap(pyd.count_by)(iteratee)

    def every(self: 'Chain[Iterable[T]]', predicate: Callable[[T], Any] | IterateeObjT=None) -> 'Chain[bool]':
        return self._wrap(pyd.every)(predicate)

    @overload
    def filter_(self: 'Chain[Mapping[T, T2]]', predicate: DictIterateeT[T2, T, bool] | IterateeObjT=None) -> 'Chain[list[T2]]':
        ...

    @overload
    def filter_(self: 'Chain[Iterable[T]]', predicate: ListIterateeT[T, bool] | IterateeObjT=None) -> 'Chain[list[T]]':
        ...

    def filter_(self, predicate=None):
        return self._wrap(pyd.filter_)(predicate)

    filter = filter_

    @overload
    def find(self: 'Chain[Mapping[T, T2]]', predicate: DictIterateeT[T2, T, bool] | IterateeObjT=None) -> 'Chain[T2 | None]':
        ...

    @overload
    def find(self: 'Chain[Iterable[T]]', predicate: ListIterateeT[T, bool] | IterateeObjT=None) -> 'Chain[T | None]':
        ...

    def find(self, predicate=None):
        return self._wrap(pyd.find)(predicate)

    @overload
    def find_last(self: 'Chain[Mapping[T, T2]]', predicate: DictIterateeT[T2, T, bool] | IterateeObjT=None) -> 'Chain[T2 | None]':
        ...

    @overload
    def find_last(self: 'Chain[Iterable[T]]', predicate: ListIterateeT[T, bool] | IterateeObjT=None) -> 'Chain[T | None]':
        ...

    def find_last(self, predicate=None):
        return self._wrap(pyd.find_last)(predicate)

    @overload
    def flat_map(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, T3]) -> 'Chain[list[T3]]':
        ...

    @overload
    def flat_map(self: 'Chain[Mapping[Any, T2]]', iteratee: None=None) -> 'Chain[list[T2]]':
        ...

    @overload
    def flat_map(self: 'Chain[Iterable[Iterable[T]]]', iteratee: ListIterateeT[T, T2]) -> 'Chain[list[T2]]':
        ...

    @overload
    def flat_map(self: 'Chain[Iterable[Iterable[T]]]', iteratee: None=None) -> 'Chain[list[T]]':
        ...

    @overload
    def flat_map(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, T2]) -> 'Chain[list[T2]]':
        ...

    @overload
    def flat_map(self: 'Chain[Iterable[T]]', iteratee: None=None) -> 'Chain[list[T]]':
        ...

    def flat_map(self, iteratee=None):
        return self._wrap(pyd.flat_map)(iteratee)

    def flat_map_deep(self, iteratee=None):
        return self._wrap(pyd.flat_map_deep)(iteratee)

    def flat_map_depth(self, iteratee=None, depth=1):
        return self._wrap(pyd.flat_map_depth)(iteratee, depth)

    @overload
    def for_each(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, Any] | IterateeObjT=None) -> 'Chain[Mapping[T, T2]]':
        ...

    @overload
    def for_each(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, Any] | IterateeObjT=None) -> 'Chain[Iterable[T]]':
        ...

    def for_each(self, iteratee=None):
        return self._wrap(pyd.for_each)(iteratee)

    @overload
    def for_each_right(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, Any] | IterateeObjT) -> 'Chain[Mapping[T, T2]]':
        ...

    @overload
    def for_each_right(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, Any] | IterateeObjT) -> 'Chain[Iterable[T]]':
        ...

    def for_each_right(self, iteratee):
        return self._wrap(pyd.for_each_right)(iteratee)

    def group_by(self, iteratee=None):
        return self._wrap(pyd.group_by)(iteratee)

    def includes(self: 'Chain[Sequence | Mapping]', target: Any, from_index: int=0) -> 'Chain[bool]':
        return self._wrap(pyd.includes)(target, from_index)

    def invoke_map(self, path, *args, **kwargs):
        return self._wrap(pyd.invoke_map)(path, *args, **kwargs)

    @overload
    def key_by(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, T3]) -> 'Chain[dict[T3, T]]':
        ...

    @overload
    def key_by(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, T2]) -> 'Chain[dict[T2, T]]':
        ...

    @overload
    def key_by(self: 'Chain[Iterable]', iteratee: IterateeObjT=None) -> 'Chain[dict]':
        ...

    def key_by(self, iteratee=None):
        return self._wrap(pyd.key_by)(iteratee)

    @overload
    def map_(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, T3]) -> 'Chain[list[T3]]':
        ...

    @overload
    def map_(self: 'Chain[Mapping]', iteratee: IterateeObjT=None) -> 'Chain[list]':
        ...

    @overload
    def map_(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, T2]) -> 'Chain[list[T2]]':
        ...

    @overload
    def map_(self: 'Chain[Iterable]', iteratee: IterateeObjT=None) -> 'Chain[list]':
        ...

    def map_(self, iteratee=None):
        return self._wrap(pyd.map_)(iteratee)

    map = map_

    def nest(self, *properties):
        return self._wrap(pyd.nest)(*properties)

    def order_by(self, keys, orders=None, reverse=False):
        return self._wrap(pyd.order_by)(keys, orders, reverse)

    @overload
    def partition(self: 'Chain[Mapping[T, T2]]', predicate: DictIterateeT[T2, T, bool]) -> 'Chain[list[list[T2]]]':
        ...

    @overload
    def partition(self: 'Chain[Mapping[Any, T2]]', predicate: IterateeObjT=None) -> 'Chain[list[list[T2]]]':
        ...

    @overload
    def partition(self: 'Chain[Iterable[T]]', predicate: ListIterateeT[T, bool]) -> 'Chain[list[list[T]]]':
        ...

    @overload
    def partition(self: 'Chain[Iterable[T]]', predicate: IterateeObjT=None) -> 'Chain[list[list[T]]]':
        ...

    def partition(self, predicate=None):
        return self._wrap(pyd.partition)(predicate)

    def pluck(self: 'Chain[Iterable]', path: str | list[str]) -> 'Chain[list]':
        return self._wrap(pyd.pluck)(path)

    def reduce_(self, iteratee=None, accumulator=None):
        return self._wrap(pyd.reduce_)(iteratee, accumulator)

    reduce = reduce_

    def reduce_right(self, iteratee=None, accumulator=None):
        return self._wrap(pyd.reduce_right)(iteratee, accumulator)

    def reductions(self, iteratee=None, accumulator=None, from_right=False):
        return self._wrap(pyd.reductions)(iteratee, accumulator, from_right)

    def reductions_right(self, iteratee=None, accumulator=None):
        return self._wrap(pyd.reductions_right)(iteratee, accumulator)

    @overload
    def reject(self: 'Chain[Mapping[T, T2]]', predicate: DictIterateeT[T2, T, bool] | IterateeObjT=None) -> 'Chain[list[T2]]':
        ...

    @overload
    def reject(self: 'Chain[Iterable[T]]', predicate: ListIterateeT[T, bool] | IterateeObjT=None) -> 'Chain[list[T]]':
        ...

    def reject(self, predicate=None):
        return self._wrap(pyd.reject)(predicate)

    def sample(self: 'Chain[Sequence[T]]') -> 'Chain[T]':
        return self._wrap(pyd.sample)()

    def sample_size(self: 'Chain[Sequence[T]]', n: int | None=None) -> 'Chain[list[T]]':
        return self._wrap(pyd.sample_size)(n)

    @overload
    def shuffle(self: 'Chain[Mapping[Any, T]]') -> 'Chain[list[T]]':
        ...

    @overload
    def shuffle(self: 'Chain[Iterable[T]]') -> 'Chain[list[T]]':
        ...

    def shuffle(self):
        return self._wrap(pyd.shuffle)()

    def size(self: 'Chain[Sized]') -> 'Chain[int]':
        return self._wrap(pyd.size)()

    def some(self: 'Chain[Iterable[T]]', predicate: Callable[[T], Any] | None=None) -> 'Chain[bool]':
        return self._wrap(pyd.some)(predicate)

    @overload
    def sort_by(self: 'Chain[Mapping[Any, T2]]', iteratee: Callable[[T2], Any] | IterateeObjT=None, reverse: bool=False) -> 'Chain[list[T2]]':
        ...

    @overload
    def sort_by(self: 'Chain[Iterable[T]]', iteratee: Callable[[T], Any] | IterateeObjT=None, reverse: bool=False) -> 'Chain[list[T]]':
        ...

    def sort_by(self, iteratee=None, reverse=False):
        return self._wrap(pyd.sort_by)(iteratee, reverse)

    def after(self: 'Chain[Callable[P, T]]', n: int) -> 'Chain[After[P, T]]':
        return self._wrap(pyd.after)(n)

    def ary(self: 'Chain[Callable[P, T]]', n: int) -> 'Chain[Ary[P, T]]':
        return self._wrap(pyd.ary)(n)

    def before(self: 'Chain[Callable[P, T]]', n: int) -> 'Chain[Before[P, T]]':
        return self._wrap(pyd.before)(n)

    def curry(self, arity=None):
        return self._wrap(pyd.curry)(arity)

    def curry_right(self, arity=None):
        return self._wrap(pyd.curry_right)(arity)

    def debounce(self: 'Chain[Callable[P, T]]', wait: int, max_wait: int | Literal[False]=False) -> 'Chain[Callable[P, T]]':
        return self._wrap(pyd.debounce)(wait, max_wait)

    def delay(self: 'Chain[Callable[P, T]]', wait: int, *args: P.args, **kwargs: P.kwargs) -> 'Chain[T]':
        return self._wrap(pyd.delay)(wait, *args, **kwargs)

    def flip(self):
        return self._wrap(pyd.flip)()

    def iterated(self):
        return self._wrap(pyd.iterated)()

    def negate(self):
        return self._wrap(pyd.negate)()

    def once(self: 'Chain[Callable[P, T]]') -> 'Chain[Once[P, T]]':
        return self._wrap(pyd.once)()

    def over_args(self, *transforms):
        return self._wrap(pyd.over_args)(*transforms)

    def partial(self, *args, **kwargs):
        return self._wrap(pyd.partial)(*args, **kwargs)

    def partial_right(self, *args, **kwargs):
        return self._wrap(pyd.partial_right)(*args, **kwargs)

    def rearg(self, *indexes):
        return self._wrap(pyd.rearg)(*indexes)

    def spread(self: 'Chain[Callable[P, T]]') -> 'Chain[Spread[P, T]]':
        return self._wrap(pyd.spread)()

    def throttle(self: 'Chain[Callable[P, T]]', wait: int) -> 'Chain[Throttle[P, T]]':
        return self._wrap(pyd.throttle)(wait)

    def unary(self: 'Chain[Callable[P, T]]') -> 'Chain[Ary[P, T]]':
        return self._wrap(pyd.unary)()

    def wrap(self, func):
        return self._wrap(pyd.wrap)(func)

    @overload
    def add(self: 'Chain[Addable[WithT, ToT]]', b: WithT) -> 'Chain[ToT]':
        ...

    @overload
    def add(self: 'Chain[WithT]', b: Addable[WithT, ToT]) -> 'Chain[ToT]':
        ...

    def add(self, b):
        return self._wrap(pyd.add)(b)

    @overload
    def sum_(self: 'Chain[Mapping[Any, Addable[T, T]]]') -> 'Chain[T | Literal[0]]':
        ...

    @overload
    def sum_(self: 'Chain[Iterable[Addable[T, T]]]') -> 'Chain[T | Literal[0]]':
        ...

    def sum_(self):
        return self._wrap(pyd.sum_)()

    sum = sum_

    @overload
    def sum_by(self: 'Chain[Mapping[T, T2]]', iteratee: DictIterateeT[T2, T, Addable[T3, T3]]) -> 'Chain[T3]':
        ...

    @overload
    def sum_by(self: 'Chain[Mapping[Any, Addable[T2, T2]]]', iteratee: None=None) -> 'Chain[T2]':
        ...

    @overload
    def sum_by(self: 'Chain[Iterable[T]]', iteratee: ListIterateeT[T, Addable[T2, T2]]) -> 'Chain[T2]':
        ...

    @overload
    def sum_by(self: 'Chain[Iterable[Addable[T, T]]]', iteratee: None=None) -> 'Chain[T]':
        ...

    def sum_by(self, iteratee=None):
        return self._wrap(pyd.sum_by)(iteratee)

    @overload
    def mean(self: 'Chain[Mapping[Any, Number]]') -> 'Chain[float]':
        ...

    @overload
    def mean(self: 'Chain[Iterable[Number]]') -> 'Chain[float]':
        ...

    def mean(self):
        return self._wrap(pyd.mean)()

    @overload
    def mean_by(self: 'Chain[Mapping[T, Number]]', iteratee: DictIterateeT[Number, T, Number] | None=None) -> 'Chain[float]':
        ...

    @overload
    def mean_by(self: 'Chain[Iterable[Number]]', iteratee: ListIterateeT[Number, Number] | None=None) -> 'Chain[float]':
        ...

    def mean_by(self, iteratee=None):
        return self._wrap(pyd.mean_by)(iteratee)

    def ceil(self: 'Chain[Number]', precision: int=0) -> 'Chain[int]':
        return self._wrap(pyd.ceil)(precision)

    def clamp(self: 'Chain[Number]', lower: Number, upper: Number | None=None) -> 'Chain[Number]':
        return self._wrap(pyd.clamp)(lower, upper)

    def divide(self: 'Chain[Number | None]', divisor: Number | None) -> 'Chain[float]':
        return self._wrap(pyd.divide)(divisor)

    def floor(self: 'Chain[Number]', precision: int=0) -> 'Chain[float]':
        return self._wrap(pyd.floor)(precision)

    @overload
    def max_(self: 'Chain[Mapping[Any, T]]', default: Unset=UNSET) -> 'Chain[T]':
        ...

    @overload
    def max_(self: 'Chain[Mapping[Any, T]]', default: T2) -> 'Chain[T | T2]':
        ...

    @overload
    def max_(self: 'Chain[Iterable[T]]', default: Unset=UNSET) -> 'Chain[T]':
        ...

    @overload
    def max_(self: 'Chain[Iterable[T]]', default: T2) -> 'Chain[T | T2]':
        ...

    def max_(self, default=UNSET):
        return self._wrap(pyd.max_)(default)

    max = max_

    def max_by(self, iteratee=None, default=UNSET):
        return self._wrap(pyd.max_by)(iteratee, default)

    def median(self, iteratee=None):
        return self._wrap(pyd.median)(iteratee)

    def min_(self, default=UNSET):
        return self._wrap(pyd.min_)(default)

    min = min_

    def min_by(self, iteratee=None, default=UNSET):
        return self._wrap(pyd.min_by)(iteratee, default)

    def moving_mean(self, size):
        return self._wrap(pyd.moving_mean)(size)

    def multiply(self, multiplicand):
        return self._wrap(pyd.multiply)(multiplicand)

    def power(self, n):
        return self._wrap(pyd.power)(n)

    def round_(self, precision=0):
        return self._wrap(pyd.round_)(precision)

    round = round_

    def scale(self, maximum=1):
        return self._wrap(pyd.scale)(maximum)

    def slope(self, point2):
        return self._wrap(pyd.slope)(point2)

    def std_deviation(self):
        return self._wrap(pyd.std_deviation)()

    def subtract(self, subtrahend):
        return self._wrap(pyd.subtract)(subtrahend)

    def transpose(self):
        return self._wrap(pyd.transpose)()

    def variance(self):
        return self._wrap(pyd.variance)()

    def zscore(self, iteratee=None):
        return self._wrap(pyd.zscore)(iteratee)

    def assign(self, *sources):
        return self._wrap(pyd.assign)(*sources)

    def assign_with(self, *sources, **kwargs):
        return self._wrap(pyd.assign_with)(*sources, **kwargs)

    def callables(self):
        return self._wrap(pyd.callables)()

    def clone(self):
        return self._wrap(pyd.clone)()

    def clone_with(self, customizer=None):
        return self._wrap(pyd.clone_with)(customizer)

    def clone_deep(self):
        return self._wrap(pyd.clone_deep)()

    def clone_deep_with(self, customizer=None):
        return self._wrap(pyd.clone_deep_with)(customizer)

    def defaults(self, *sources):
        return self._wrap(pyd.defaults)(*sources)

    def defaults_deep(self, *sources):
        return self._wrap(pyd.defaults_deep)(*sources)

    def find_key(self, predicate=None):
        return self._wrap(pyd.find_key)(predicate)

    def find_last_key(self, predicate=None):
        return self._wrap(pyd.find_last_key)(predicate)

    def for_in(self, iteratee=None):
        return self._wrap(pyd.for_in)(iteratee)

    def for_in_right(self, iteratee=None):
        return self._wrap(pyd.for_in_right)(iteratee)

    @overload
    def get(self: 'Chain[Sequence[T]]', path: SupportsInt, default: T2) -> 'Chain[T | T2]':
        ...

    @overload
    def get(self: 'Chain[Sequence[T]]', path: SupportsInt, default: None=None) -> 'Chain[T | None]':
        ...

    @overload
    def get(self: 'Chain[Any]', path: Any, default: Any=None) -> 'Chain[Any]':
        ...

    def get(self, path, default=None):
        return self._wrap(pyd.get)(path, default)

    def has(self, path):
        return self._wrap(pyd.has)(path)

    def invert(self):
        return self._wrap(pyd.invert)()

    def invert_by(self, iteratee=None):
        return self._wrap(pyd.invert_by)(iteratee)

    def invoke(self, path, *args, **kwargs):
        return self._wrap(pyd.invoke)(path, *args, **kwargs)

    def keys(self):
        return self._wrap(pyd.keys)()

    def map_keys(self, iteratee=None):
        return self._wrap(pyd.map_keys)(iteratee)

    def map_values(self, iteratee=None):
        return self._wrap(pyd.map_values)(iteratee)

    def map_values_deep(self, iteratee=None, property_path=UNSET):
        return self._wrap(pyd.map_values_deep)(iteratee, property_path)

    def merge(self, *sources):
        return self._wrap(pyd.merge)(*sources)

    def merge_with(self, *sources, **kwargs):
        return self._wrap(pyd.merge_with)(*sources, **kwargs)

    def omit(self, *properties):
        return self._wrap(pyd.omit)(*properties)

    def omit_by(self, iteratee=None):
        return self._wrap(pyd.omit_by)(iteratee)

    def parse_int(self, radix=None):
        return self._wrap(pyd.parse_int)(radix)

    def pick(self, *properties):
        return self._wrap(pyd.pick)(*properties)

    def pick_by(self, iteratee=None):
        return self._wrap(pyd.pick_by)(iteratee)

    def rename_keys(self, key_map):
        return self._wrap(pyd.rename_keys)(key_map)

    def set_(self, path, value):
        return self._wrap(pyd.set_)(path, value)

    set = set_

    def set_with(self, path, value, customizer=None):
        return self._wrap(pyd.set_with)(path, value, customizer)

    def to_boolean(self, true_values=('true', '1'), false_values=('false', '0')):
        return self._wrap(pyd.to_boolean)(true_values, false_values)

    def to_dict(self):
        return self._wrap(pyd.to_dict)()

    def to_integer(self):
        return self._wrap(pyd.to_integer)()

    def to_list(self, split_strings=True):
        return self._wrap(pyd.to_list)(split_strings)

    def to_number(self, precision=0):
        return self._wrap(pyd.to_number)(precision)

    def to_pairs(self):
        return self._wrap(pyd.to_pairs)()

    def to_string(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.to_string)()

    def transform(self, iteratee=None, accumulator=None):
        return self._wrap(pyd.transform)(iteratee, accumulator)

    def update(self, path, updater):
        return self._wrap(pyd.update)(path, updater)

    def update_with(self, path, updater, customizer=None):
        return self._wrap(pyd.update_with)(path, updater, customizer)

    def unset(self, path):
        return self._wrap(pyd.unset)(path)

    def values(self):
        return self._wrap(pyd.values)()

    def eq(self: 'Chain[Any]', other: T) -> 'Chain[bool]':
        return self._wrap(pyd.eq)(other)

    def gt(self: 'Chain[Any]', other: Any) -> 'Chain[bool]':
        return self._wrap(pyd.gt)(other)

    def gte(self: 'Chain[Any]', other: Any) -> 'Chain[bool]':
        return self._wrap(pyd.gte)(other)

    def lt(self: 'Chain[Any]', other: Any) -> 'Chain[bool]':
        return self._wrap(pyd.lt)(other)

    def lte(self: 'Chain[Any]', other: Any) -> 'Chain[bool]':
        return self._wrap(pyd.lte)(other)

    def in_range(self: 'Chain[Number]', start: Number=0, end: Number | None=None) -> 'Chain[bool]':
        return self._wrap(pyd.in_range)(start, end)

    def is_associative(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_associative)()

    def is_blank(self: 'Chain[str]') -> 'Chain[bool]':
        return self._wrap(pyd.is_blank)()

    def is_boolean(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_boolean)()

    def is_builtin(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_builtin)()

    def is_date(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_date)()

    def is_decreasing(self: 'Chain[list]') -> 'Chain[bool]':
        return self._wrap(pyd.is_decreasing)()

    def is_dict(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_dict)()

    def is_empty(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_empty)()

    def is_equal(self: 'Chain[Any]', other: Any) -> 'Chain[bool]':
        return self._wrap(pyd.is_equal)(other)

    def is_equal_with(self: 'Chain[list[T] | dict[Any, T] | Any]', other: list[T2] | dict[Any, T2] | Any, customizer: Callable[[T, T2], bool | None] | None) -> 'Chain[bool]':
        return self._wrap(pyd.is_equal_with)(other, customizer)

    def is_error(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_error)()

    def is_even(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_even)()

    def is_float(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_float)()

    def is_function(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_function)()

    def is_increasing(self: 'Chain[list]') -> 'Chain[bool]':
        return self._wrap(pyd.is_increasing)()

    def is_indexed(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_indexed)()

    def is_instance_of(self: 'Chain[Any]', types: type | tuple[type]) -> 'Chain[bool]':
        return self._wrap(pyd.is_instance_of)(types)

    def is_integer(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_integer)()

    def is_iterable(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_iterable)()

    def is_json(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_json)()

    def is_list(self: 'Chain[list[T] | Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_list)()

    def is_match(self: 'Chain[Any]', source: Any) -> 'Chain[bool]':
        return self._wrap(pyd.is_match)(source)

    def is_match_with(self, source, customizer=None, _key=UNSET, _obj=UNSET, _source=UNSET):
        return self._wrap(pyd.is_match_with)(source, customizer, _key, _obj, _source)

    def is_monotone(self: 'Chain[list[T] | T]', op: Callable[[T, T], bool]) -> 'Chain[bool]':
        return self._wrap(pyd.is_monotone)(op)

    def is_nan(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_nan)()

    def is_negative(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_negative)()

    def is_none(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_none)()

    def is_number(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_number)()

    def is_object(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_object)()

    def is_odd(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_odd)()

    def is_positive(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_positive)()

    def is_reg_exp(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_reg_exp)()

    def is_set(self: 'Chain[set[T] | Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_set)()

    def is_strictly_decreasing(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_strictly_decreasing)()

    def is_strictly_increasing(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_strictly_increasing)()

    def is_string(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_string)()

    def is_tuple(self: 'Chain[tuple[T] | Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_tuple)()

    def is_zero(self: 'Chain[Any]') -> 'Chain[bool]':
        return self._wrap(pyd.is_zero)()

    def camel_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.camel_case)()

    def capitalize(self: 'Chain[Representable]', strict: bool=True) -> 'Chain[str]':
        return self._wrap(pyd.capitalize)(strict)

    def chars(self: 'Chain[Representable]') -> 'Chain[list[str]]':
        return self._wrap(pyd.chars)()

    def chop(self: 'Chain[Representable]', step: int) -> 'Chain[list[str]]':
        return self._wrap(pyd.chop)(step)

    def chop_right(self: 'Chain[Representable]', step: int) -> 'Chain[list[str]]':
        return self._wrap(pyd.chop_right)(step)

    def clean(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.clean)()

    def count_substr(self: 'Chain[Representable]', subtext: Representable) -> 'Chain[int]':
        return self._wrap(pyd.count_substr)(subtext)

    def deburr(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.deburr)()

    def decapitalize(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.decapitalize)()

    def ends_with(self: 'Chain[Representable]', target: Representable, position: int | None=None) -> 'Chain[bool]':
        return self._wrap(pyd.ends_with)(target, position)

    def ensure_ends_with(self: 'Chain[Representable]', suffix: Representable) -> 'Chain[str]':
        return self._wrap(pyd.ensure_ends_with)(suffix)

    def ensure_starts_with(self: 'Chain[Representable]', prefix: Representable) -> 'Chain[str]':
        return self._wrap(pyd.ensure_starts_with)(prefix)

    def escape(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.escape)()

    def escape_reg_exp(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.escape_reg_exp)()

    def has_substr(self: 'Chain[Representable]', subtext: Representable) -> 'Chain[bool]':
        return self._wrap(pyd.has_substr)(subtext)

    def human_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.human_case)()

    def insert_substr(self: 'Chain[Representable]', index: int, subtext: Representable) -> 'Chain[str]':
        return self._wrap(pyd.insert_substr)(index, subtext)

    def join(self: 'Chain[list[Representable]]', separator: Representable='') -> 'Chain[str]':
        return self._wrap(pyd.join)(separator)

    def kebab_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.kebab_case)()

    def lines(self: 'Chain[Representable]') -> 'Chain[list[str]]':
        return self._wrap(pyd.lines)()

    def lower_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.lower_case)()

    def lower_first(self: 'Chain[str]') -> 'Chain[str]':
        return self._wrap(pyd.lower_first)()

    def number_format(self: 'Chain[int | float]', scale: int=0, decimal_separator: str='.', order_separator: str=',') -> 'Chain[str]':
        return self._wrap(pyd.number_format)(scale, decimal_separator, order_separator)

    def pad(self: 'Chain[Representable]', length: int, chars: str=' ') -> 'Chain[str]':
        return self._wrap(pyd.pad)(length, chars)

    def pad_end(self: 'Chain[Representable]', length: int, chars: str=' ') -> 'Chain[str]':
        return self._wrap(pyd.pad_end)(length, chars)

    def pad_start(self: 'Chain[Representable]', length: int, chars: str=' ') -> 'Chain[str]':
        return self._wrap(pyd.pad_start)(length, chars)

    def pascal_case(self: 'Chain[Representable]', strict: bool=True) -> 'Chain[str]':
        return self._wrap(pyd.pascal_case)(strict)

    def predecessor(self: 'Chain[str]') -> 'Chain[str]':
        return self._wrap(pyd.predecessor)()

    def prune(self: 'Chain[Representable]', length: int=0, omission: str='...') -> 'Chain[str]':
        return self._wrap(pyd.prune)(length, omission)

    def quote(self: 'Chain[Representable]', quote_char: Representable='"') -> 'Chain[str]':
        return self._wrap(pyd.quote)(quote_char)

    def reg_exp_js_match(self: 'Chain[Representable]', reg_exp: str) -> 'Chain[list[str]]':
        return self._wrap(pyd.reg_exp_js_match)(reg_exp)

    def reg_exp_js_replace(self: 'Chain[Representable]', reg_exp: str, repl: Representable | Callable[[re.Match[str]], str]) -> 'Chain[str]':
        return self._wrap(pyd.reg_exp_js_replace)(reg_exp, repl)

    def reg_exp_replace(self: 'Chain[Representable]', pattern: Representable | Pattern, repl: Representable | Callable[[re.Match[str]], str], ignore_case: bool=False, count: int=0) -> 'Chain[str]':
        return self._wrap(pyd.reg_exp_replace)(pattern, repl, ignore_case, count)

    def repeat(self: 'Chain[Representable]', n: int=0) -> 'Chain[str]':
        return self._wrap(pyd.repeat)(n)

    def replace(self: 'Chain[Representable]', pattern: Representable | Pattern, repl: Representable | Callable[[re.Match[str]], str], ignore_case: bool=False, count: int=0, escape: bool=True, from_start: bool=False, from_end: bool=False) -> 'Chain[str]':
        return self._wrap(pyd.replace)(pattern, repl, ignore_case, count, escape, from_start, from_end)

    def replace_end(self: 'Chain[Representable]', pattern: Representable | Pattern, repl: Representable | Callable[[re.Match[str]], str], ignore_case: bool=False, escape: bool=True) -> 'Chain[str]':
        return self._wrap(pyd.replace_end)(pattern, repl, ignore_case, escape)

    def replace_start(self: 'Chain[Representable]', pattern: Representable | Pattern, repl: Representable | Callable[[re.Match[str]], str], ignore_case: bool=False, escape: bool=True) -> 'Chain[str]':
        return self._wrap(pyd.replace_start)(pattern, repl, ignore_case, escape)

    def separator_case(self: 'Chain[Representable]', separator: str) -> 'Chain[str]':
        return self._wrap(pyd.separator_case)(separator)

    def series_phrase(self: 'Chain[list[Representable]]', separator: Representable=', ', last_separator: Representable=' and ', serial: bool=False) -> 'Chain[str]':
        return self._wrap(pyd.series_phrase)(separator, last_separator, serial)

    def series_phrase_serial(self: 'Chain[list[Representable]]', separator: Representable=', ', last_separator: Representable=' and ') -> 'Chain[str]':
        return self._wrap(pyd.series_phrase_serial)(separator, last_separator)

    def slugify(self: 'Chain[Representable]', separator: str='-') -> 'Chain[str]':
        return self._wrap(pyd.slugify)(separator)

    def snake_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.snake_case)()

    def split(self: 'Chain[Representable]', separator: Any=UNSET) -> 'Chain[list[str]]':
        return self._wrap(pyd.split)(separator)

    def start_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.start_case)()

    def starts_with(self: 'Chain[Representable]', target: Representable, position: int=0) -> 'Chain[bool]':
        return self._wrap(pyd.starts_with)(target, position)

    def strip_tags(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.strip_tags)()

    def substr_left(self: 'Chain[Representable]', subtext: str) -> 'Chain[str]':
        return self._wrap(pyd.substr_left)(subtext)

    def substr_left_end(self: 'Chain[Representable]', subtext: str) -> 'Chain[str]':
        return self._wrap(pyd.substr_left_end)(subtext)

    def substr_right(self: 'Chain[Representable]', subtext: str) -> 'Chain[str]':
        return self._wrap(pyd.substr_right)(subtext)

    def substr_right_end(self: 'Chain[Representable]', subtext: str) -> 'Chain[str]':
        return self._wrap(pyd.substr_right_end)(subtext)

    def successor(self: 'Chain[str]') -> 'Chain[str]':
        return self._wrap(pyd.successor)()

    def surround(self: 'Chain[Representable]', wrapper: Representable) -> 'Chain[str]':
        return self._wrap(pyd.surround)(wrapper)

    def swap_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.swap_case)()

    def title_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.title_case)()

    def to_lower(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.to_lower)()

    def to_upper(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.to_upper)()

    def trim(self: 'Chain[Representable]', chars: str | None=None) -> 'Chain[str]':
        return self._wrap(pyd.trim)(chars)

    def trim_end(self: 'Chain[Representable]', chars: str | None=None) -> 'Chain[str]':
        return self._wrap(pyd.trim_end)(chars)

    def trim_start(self: 'Chain[Representable]', chars: str | None=None) -> 'Chain[str]':
        return self._wrap(pyd.trim_start)(chars)

    def truncate(self: 'Chain[Representable]', length: int=30, omission: str='...', separator: str | Pattern | None=None) -> 'Chain[str]':
        return self._wrap(pyd.truncate)(length, omission, separator)

    def unescape(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.unescape)()

    def upper_case(self: 'Chain[Representable]') -> 'Chain[str]':
        return self._wrap(pyd.upper_case)()

    def upper_first(self: 'Chain[str]') -> 'Chain[str]':
        return self._wrap(pyd.upper_first)()

    def unquote(self: 'Chain[Representable]', quote_char: Representable='"') -> 'Chain[str]':
        return self._wrap(pyd.unquote)(quote_char)

    def words(self: 'Chain[Representable]', pattern: str | None=None) -> 'Chain[list[str]]':
        return self._wrap(pyd.words)(pattern)

    def attempt(self: 'Chain[Callable[P, T]]', *args: P.args, **kwargs: P.kwargs) -> 'Chain[T | Exception]':
        return self._wrap(pyd.attempt)(*args, **kwargs)

    def cond(self, *extra_pairs):
        return self._wrap(pyd.cond)(*extra_pairs)

    def conforms(self):
        return self._wrap(pyd.conforms)()

    def conforms_to(self, source):
        return self._wrap(pyd.conforms_to)(source)

    def constant(self):
        return self._wrap(pyd.constant)()

    def default_to(self, default_value):
        return self._wrap(pyd.default_to)(default_value)

    def default_to_any(self, *default_values):
        return self._wrap(pyd.default_to_any)(*default_values)

    @overload
    def iteratee(self: 'Chain[Callable[P, T]]') -> 'Chain[Callable[P, T]]':
        ...

    @overload
    def iteratee(self: 'Chain[dict]') -> 'Chain[Callable[[IterateeObjT], bool]]':
        ...

    @overload
    def iteratee(self: 'Chain[IterateeObjT]') -> 'Chain[Callable[[IterateeObjT], Any]]':
        ...

    @overload
    def iteratee(self: 'Chain[Any]') -> 'Chain[Callable[[T], T]]':
        ...

    def iteratee(self):
        return self._wrap(pyd.iteratee)()

    def matches(self: 'Chain[IterateeObjT]') -> 'Chain[Callable[[IterateeObjT], bool]]':
        return self._wrap(pyd.matches)()

    def matches_property(self: 'Chain[str]', value: IterateeObjT) -> 'Chain[Callable[[IterateeObjT], bool]]':
        return self._wrap(pyd.matches_property)(value)

    def memoize(self, resolver=None):
        return self._wrap(pyd.memoize)(resolver)

    def method(self, *args, **kwargs):
        return self._wrap(pyd.method)(*args, **kwargs)

    def method_of(self, *args, **kwargs):
        return self._wrap(pyd.method_of)(*args, **kwargs)

    def over(self):
        return self._wrap(pyd.over)()

    def over_every(self):
        return self._wrap(pyd.over_every)()

    def over_some(self):
        return self._wrap(pyd.over_some)()

    def property_(self: 'Chain[str | Iterable[str]]') -> 'Chain[Callable[[IterateeObjT], Any]]':
        return self._wrap(pyd.property_)()

    property = property_

    def property_of(self):
        return self._wrap(pyd.property_of)()

    def random(self=0, stop=1, floating=False):
        return self._wrap(pyd.random)(stop, floating)

    def result(self, key, default=None):
        return self._wrap(pyd.result)(key, default)

    def retry(self=3, delay=0.5, max_delay=150.0, scale=2.0, jitter=0, exceptions=(Exception,), on_exception=None):
        return self._wrap(pyd.retry)(delay, max_delay, scale, jitter, exceptions, on_exception)

    def times(self, iteratee=None):
        return self._wrap(pyd.times)(iteratee)

    def to_path(self):
        return self._wrap(pyd.to_path)()

