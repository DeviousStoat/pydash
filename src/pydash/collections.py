"""
Functions that operate on lists and dicts.

.. versionadded:: 1.0.0
"""

from functools import cmp_to_key
from typing import Any, Callable, Generator, Mapping, Iterable, Sequence, Sized, overload
import random

import pydash as pyd

from .helpers import (
    callit,
    cmp,
    getargcount,
    iterator,
    iteriteratee,
)
from .types import (
    T,
    T2,
    T3,
    IterateeObjT,
    CollectionT,
    DictIterateeT,
    ListIterateeT,
)


__all__ = (
    "at",
    "count_by",
    "every",
    "filter_",
    "find",
    "find_last",
    "flat_map",
    "flat_map_deep",
    "flat_map_depth",
    "for_each",
    "for_each_right",
    "group_by",
    "includes",
    "invoke_map",
    "key_by",
    "map_",
    "nest",
    "order_by",
    "partition",
    "pluck",
    "reduce_",
    "reduce_right",
    "reductions",
    "reductions_right",
    "reject",
    "sample",
    "sample_size",
    "shuffle",
    "size",
    "some",
    "sort_by",
)


@overload
def at(collection: Mapping[T, T2], *paths: T | Iterable[T]) -> list[T2]:
    ...


@overload
def at(collection: Iterable[T], *paths: int | Iterable[int]) -> list[T]:
    ...


def at(collection, *paths):
    """
    Creates a list of elements from the specified indexes, or keys, of the collection. Indexes may
    be specified as individual arguments or as arrays of indexes.

    Args:
        collection (list|dict): Collection to iterate over.
        *paths (mixed): The indexes of `collection` to retrieve, specified as individual indexes or
            arrays of indexes.

    Returns:
        list: filtered list

    Example:

        >>> at([1, 2, 3, 4], 0, 2)
        [1, 3]
        >>> at({'a': 1, 'b': 2, 'c': 3, 'd': 4}, 'a', 'c')
        [1, 3]
        >>> at({'a': 1, 'b': 2, 'c': {'d': {'e': 3}}}, 'a', ['c', 'd', 'e'])
        [1, 3]

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.1.0
        Support deep path access.
    """
    return pyd.properties(*paths)(collection)


@overload
def count_by(collection: Mapping[Any, T2], iteratee: None = None) -> dict[T2, int]:
    ...


@overload
def count_by(collection: Mapping[T, T2], iteratee: DictIterateeT[T2, T, T3]) -> dict[T3, int]:
    ...


@overload
def count_by(collection: Iterable[T], iteratee: None = None) -> dict[T, int]:
    ...


@overload
def count_by(collection: Iterable[T], iteratee: ListIterateeT[T, T2]) -> dict[T2, int]:
    ...


def count_by(collection, iteratee=None):
    """
    Creates an object composed of keys generated from the results of running each element of
    `collection` through the iteratee.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        dict: Dict containing counts by key.

    Example:

        >>> results = count_by([1, 2, 1, 2, 3, 4])
        >>> assert results == {1: 2, 2: 2, 3: 1, 4: 1}
        >>> results = count_by(['a', 'A', 'B', 'b'], lambda x: x.lower())
        >>> assert results == {'a': 2, 'b': 2}
        >>> results = count_by({'a': 1, 'b': 1, 'c': 3, 'd': 3})
        >>> assert results == {1: 2, 3: 2}

    .. versionadded:: 1.0.0
    """
    ret = {}

    for result in iteriteratee(collection, iteratee):
        ret.setdefault(result[0], 0)
        ret[result[0]] += 1

    return ret


def every(collection: Iterable[T], predicate: Callable[[T], Any] | IterateeObjT = None) -> bool:
    """
    Checks if the predicate returns a truthy value for all elements of a collection.
    # TODO: this below is not true
    The predicate is invoked with three arguments: ``(value, index|key, collection)``.
    If a property name is passed for predicate, the created :func:`pluck` style
    predicate will return the property value of the given element.
    If an object is passed for predicate, the created :func:`.matches` style
    predicate will return ``True`` for elements that have the properties of the given object, else
    ``False``.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        bool: Whether all elements are truthy.

    Example:

        >>> every([1, True, 'hello'])
        True
        >>> every([1, False, 'hello'])
        False
        >>> every([{'a': 1}, {'a': True}, {'a': 'hello'}], 'a')
        True
        >>> every([{'a': 1}, {'a': False}, {'a': 'hello'}], 'a')
        False
        >>> every([{'a': 1}, {'a': 1}], {'a': 1})
        True
        >>> every([{'a': 1}, {'a': 2}], {'a': 1})
        False

    .. versionadded:: 1.0.0

    .. versionchanged: 4.0.0
        Removed alias ``all_``.
    """
    if predicate:
        cbk = pyd.iteratee(predicate)
        collection = (cbk(item) for item in collection)

    return all(collection)


@overload
def filter_(
    collection: Mapping[T, T2], predicate: DictIterateeT[T2, T, bool] | IterateeObjT = None
) -> list[T2]:
    ...


@overload
def filter_(
    collection: Iterable[T], predicate: ListIterateeT[T, bool] | IterateeObjT = None
) -> list[T]:
    ...


def filter_(collection, predicate=None):
    """
    Iterates over elements of a collection, returning a list of all elements the predicate returns
    truthy for.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        list: Filtered list.

    Example:

        >>> results = filter_([{'a': 1}, {'b': 2}, {'a': 1, 'b': 3}], {'a': 1})
        >>> assert results == [{'a': 1}, {'a': 1, 'b': 3}]
        >>> filter_([1, 2, 3, 4], lambda x: x >= 3)
        [3, 4]

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed alias ``select``.
    """
    return [value for is_true, value, _, _ in iteriteratee(collection, predicate) if is_true]


@overload
def find(
    collection: Mapping[T, T2], predicate: DictIterateeT[T2, T, bool] | IterateeObjT = None
) -> T2 | None:
    ...


@overload
def find(
    collection: Iterable[T], predicate: ListIterateeT[T, bool] | IterateeObjT = None
) -> T | None:
    ...


def find(collection, predicate=None):
    """
    Iterates over elements of a collection, returning the first element that the predicate returns
    truthy for.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        mixed: First element found or ``None``.

    Example:

        >>> find([1, 2, 3, 4], lambda x: x >= 3)
        3
        >>> find([{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}], {'a': 1})
        {'a': 1}

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed aliases ``detect`` and ``find_where``.
    """
    search = (value for is_true, value, _, _ in iteriteratee(collection, predicate) if is_true)
    return next(search, None)


@overload
def find_last(
    collection: Mapping[T, T2], predicate: DictIterateeT[T2, T, bool] | IterateeObjT = None
) -> T2 | None:
    ...


@overload
def find_last(
    collection: Iterable[T], predicate: ListIterateeT[T, bool] | IterateeObjT = None
) -> T | None:
    ...


def find_last(collection, predicate=None):
    """
    This method is like :func:`find` except that it iterates over elements of a `collection` from
    right to left.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        mixed: Last element found or ``None``.

    Example:

        >>> find_last([1, 2, 3, 4], lambda x: x >= 3)
        4
        >>> results = find_last([{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}],\
                                 {'a': 1})
        >>> assert results == {'a': 1, 'b': 2}

    .. versionadded:: 1.0.0
    """
    search = (
        value
        for is_true, value, _, _ in iteriteratee(collection, predicate, reverse=True)
        if is_true
    )
    return next(search, None)


@overload
def flat_map(collection: Mapping[T, T2], iteratee: DictIterateeT[T2, T, T3]) -> list[T3]:
    ...


@overload
def flat_map(collection: Mapping[Any, T2], iteratee: None = None) -> list[T2]:
    ...


@overload
def flat_map(collection: Iterable[Iterable[T]], iteratee: ListIterateeT[T, T2]) -> list[T2]:
    ...


@overload
def flat_map(collection: Iterable[Iterable[T]], iteratee: None = None) -> list[T]:
    ...


@overload
def flat_map(collection: Iterable[T], iteratee: ListIterateeT[T, T2]) -> list[T2]:
    ...


@overload
def flat_map(collection: Iterable[T], iteratee: None = None) -> list[T]:
    ...


# TODO: add typing for object iteratee
def flat_map(collection, iteratee=None):
    """
    Creates a flattened list of values by running each element in collection thru `iteratee` and
    flattening the mapped results. The `iteratee` is invoked with three arguments: ``(value,
    index|key, collection)``.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        list: Flattened mapped list.

    Example:

        >>> duplicate = lambda n: [[n, n]]
        >>> flat_map([1, 2], duplicate)
        [[1, 1], [2, 2]]

    .. versionadded:: 4.0.0
    """
    return pyd.flatten(itermap(collection, iteratee=iteratee))


# TODO: too deep :(
def flat_map_deep(collection, iteratee=None):
    """
    This method is like :func:`flat_map` except that it recursively flattens the mapped results.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        list: Flattened mapped list.

    Example:

        >>> duplicate = lambda n: [[n, n]]
        >>> flat_map_deep([1, 2], duplicate)
        [1, 1, 2, 2]

    .. versionadded:: 4.0.0
    """
    return pyd.flatten_deep(itermap(collection, iteratee=iteratee))


# TODO: too deep :(
def flat_map_depth(collection, iteratee=None, depth=1):
    """
    This method is like :func:`flat_map` except that it recursively flattens the mapped results up
    to `depth` times.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        list: Flattened mapped list.

    Example:

        >>> duplicate = lambda n: [[n, n]]
        >>> flat_map_depth([1, 2], duplicate, 1)
        [[1, 1], [2, 2]]
        >>> flat_map_depth([1, 2], duplicate, 2)
        [1, 1, 2, 2]

    .. versionadded:: 4.0.0
    """
    return pyd.flatten_depth(itermap(collection, iteratee=iteratee), depth=depth)


@overload
def for_each(
    collection: Mapping[T, T2], iteratee: DictIterateeT[T2, T, Any] | IterateeObjT = None
) -> Mapping[T, T2]:
    ...


@overload
def for_each(
    collection: Iterable[T], iteratee: ListIterateeT[T, Any] | IterateeObjT = None
) -> Iterable[T]:
    ...


def for_each(collection, iteratee=None):
    """
    Iterates over elements of a collection, executing the iteratee for each element.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        list|dict: `collection`

    Example:

        >>> results = {}
        >>> def cb(x): results[x] = x ** 2
        >>> for_each([1, 2, 3, 4], cb)
        [1, 2, 3, 4]
        >>> assert results == {1: 1, 2: 4, 3: 9, 4: 16}

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed alias ``each``.
    """
    next((None for ret, _, _, _ in iteriteratee(collection, iteratee) if ret is False), None)
    return collection


@overload
def for_each_right(
    collection: Mapping[T, T2], iteratee: DictIterateeT[T2, T, Any] | IterateeObjT
) -> Mapping[T, T2]:
    ...


@overload
def for_each_right(
    collection: Iterable[T], iteratee: ListIterateeT[T, Any] | IterateeObjT
) -> Iterable[T]:
    ...


def for_each_right(collection, iteratee):
    """
    This method is like :func:`for_each` except that it iterates over elements of a `collection`
    from right to left.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        list|dict: `collection`

    Example:

        >>> results = {'total': 1}
        >>> def cb(x): results['total'] = x * results['total']
        >>> for_each_right([1, 2, 3, 4], cb)
        [1, 2, 3, 4]
        >>> assert results == {'total': 24}

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed alias ``each_right``.
    """
    next(
        (None for ret, _, _, _ in iteriteratee(collection, iteratee, reverse=True) if ret is False),
        None,
    )
    return collection


# TODO: tricky
def group_by(collection, iteratee=None):
    """
    Creates an object composed of keys generated from the results of running each element of a
    `collection` through the iteratee.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        dict: Results of grouping by `iteratee`.

    Example:

        >>> results = group_by([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], 'a')
        >>> assert results == {1: [{'a': 1, 'b': 2}], 3: [{'a': 3, 'b': 4}]}
        >>> results = group_by([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], {'a': 1})
        >>> assert results == {False: [{'a': 3, 'b': 4}],\
                               True: [{'a': 1, 'b': 2}]}

    .. versionadded:: 1.0.0
    """
    ret = {}
    cbk = pyd.iteratee(iteratee)

    for value in collection:
        key = cbk(value)
        ret.setdefault(key, [])
        ret[key].append(value)

    return ret


def includes(collection: Sequence | Mapping, target: Any, from_index: int = 0) -> bool:
    """
    Checks if a given value is present in a collection. If `from_index` is negative, it is used as
    the offset from the end of the collection.

    Args:
        collection (list|dict): Collection to iterate over.
        target (mixed): Target value to compare to.
        from_index (int, optional): Offset to start search from.

    Returns:
        bool: Whether `target` is in `collection`.

    Example:

        >>> includes([1, 2, 3, 4], 2)
        True
        >>> includes([1, 2, 3, 4], 2, from_index=2)
        False
        >>> includes({'a': 1, 'b': 2, 'c': 3, 'd': 4}, 2)
        True

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Renamed from ``contains`` to ``includes`` and removed alias
        ``include``.
    """
    if isinstance(collection, dict):
        collection_values = collection.values()
    else:
        # only makes sense to do this if `collection` is not a dict
        collection_values = collection[from_index:]

    return target in collection_values


# TODO: invoke the map yourself :(
def invoke_map(collection, path, *args, **kwargs):
    """
    Invokes the method at `path` of each element in `collection`, returning a list of the results of
    each invoked method. Any additional arguments are provided to each invoked method. If `path` is
    a function, it's invoked for each element in `collection`.

    Args:
        collection (list|dict): Collection to iterate over.
        path (str|func): String path to method to invoke or callable to invoke for each element in
            `collection`.
        args (optional): Arguments to pass to method call.
        kwargs (optional): Keyword arguments to pass to method call.

    Returns:
        list: List of results of invoking method of each item.

    Example:

        >>> items = [{'a': [{'b': 1}]}, {'a': [{'c': 2}]}]
        >>> expected = [{'b': 1}.items(), {'c': 2}.items()]
        >>> invoke_map(items, 'a[0].items') == expected
        True

    .. versionadded:: 4.0.0
    """
    return map_(collection, lambda item: pyd.invoke(item, path, *args, **kwargs))


@overload
def key_by(collection: Mapping[T, T2], iteratee: DictIterateeT[T2, T, T3]) -> dict[T3, T]:
    ...


@overload
def key_by(collection: Iterable[T], iteratee: ListIterateeT[T, T2]) -> dict[T2, T]:
    ...


@overload
def key_by(collection: Iterable, iteratee: IterateeObjT = None) -> dict:
    ...


def key_by(collection, iteratee=None):
    """
    Creates an object composed of keys generated from the results of running each element of the
    collection through the given iteratee.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        dict: Results of indexing by `iteratee`.

    Example:

        >>> results = key_by([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], 'a')
        >>> assert results == {1: {'a': 1, 'b': 2}, 3: {'a': 3, 'b': 4}}


    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Renamed from ``index_by`` to ``key_by``.
    """
    ret = {}
    cbk = pyd.iteratee(iteratee)

    for value in collection:
        ret[cbk(value)] = value

    return ret


@overload
def map_(collection: Mapping[T, T2], iteratee: DictIterateeT[T2, T, T3]) -> list[T3]:
    ...


@overload
def map_(collection: Mapping, iteratee: IterateeObjT = None) -> list:
    ...


@overload
def map_(collection: Iterable[T], iteratee: ListIterateeT[T, T2]) -> list[T2]:
    ...


@overload
def map_(collection: Iterable, iteratee: IterateeObjT = None) -> list:
    ...


def map_(collection, iteratee=None):
    """
    Creates an array of values by running each element in the collection through the iteratee. The
    iteratee is invoked with three arguments: ``(value, index|key, collection)``. If a property name
    is passed for iteratee, the created :func:`pluck` style iteratee will return the property value
    of the given element. If an object is passed for iteratee, the created :func:`.matches` style
    iteratee will return ``True`` for elements that have the properties of the given object, else
    ``False``.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.

    Returns:
        list: Mapped list.

    Example:

        >>> map_([1, 2, 3, 4], str)
        ['1', '2', '3', '4']
        >>> map_([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}], 'a')
        [1, 3, 5]
        >>> map_([[[0, 1]], [[2, 3]], [[4, 5]]], '0.1')
        [1, 3, 5]
        >>> map_([{'a': {'b': 1}}, {'a': {'b': 2}}], 'a.b')
        [1, 2]
        >>> map_([{'a': {'b': [0, 1]}}, {'a': {'b': [2, 3]}}], 'a.b[1]')
        [1, 3]

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed alias ``collect``.
    """
    return list(itermap(collection, iteratee))


# TODO: tricky
def nest(collection, *properties):
    """
    This method is like :func:`group_by` except that it supports nested grouping by multiple string
    `properties`. If only a single key is given, it is like calling ``group_by(collection, prop)``.

    Args:
        collection (list|dict): Collection to iterate over.
        *properties (str): Properties to nest by.

    Returns:
        dict: Results of nested grouping by `properties`.

    Example:

        >>> results = nest([{'shape': 'square', 'color': 'red', 'qty': 5},\
                            {'shape': 'square', 'color': 'blue', 'qty': 10},\
                            {'shape': 'square', 'color': 'orange', 'qty': 5},\
                            {'shape': 'circle', 'color': 'yellow', 'qty': 5},\
                            {'shape': 'circle', 'color': 'pink', 'qty': 10},\
                            {'shape': 'oval', 'color': 'purple', 'qty': 5}],\
                           'shape', 'qty')
        >>> expected = {\
            'square': {5: [{'shape': 'square', 'color': 'red', 'qty': 5},\
                           {'shape': 'square', 'color': 'orange', 'qty': 5}],\
                       10: [{'shape': 'square', 'color': 'blue', 'qty': 10}]},\
            'circle': {5: [{'shape': 'circle', 'color': 'yellow', 'qty': 5}],\
                       10: [{'shape': 'circle', 'color': 'pink', 'qty': 10}]},\
            'oval': {5: [{'shape': 'oval', 'color': 'purple', 'qty': 5}]}}
        >>> results == expected
        True

    .. versionadded:: 4.3.0
    """
    if not properties:
        return collection

    properties = pyd.flatten(properties)
    first, rest = properties[0], properties[1:]

    return pyd.map_values(group_by(collection, first), lambda value: nest(value, *rest))


# TODO: see you later, mayhem it is
def order_by(collection, keys, orders=None, reverse=False):
    """
    This method is like :func:`sort_by` except that it sorts by key names instead of an iteratee
    function. Keys can be sorted in descending order by prepending a ``"-"`` to the key name (e.g.
    ``"name"`` would become ``"-name"``) or by passing a list of boolean sort options via `orders`
    where ``True`` is ascending and ``False`` is descending.

    Args:
        collection (list|dict): Collection to iterate over.
        keys (list): List of keys to sort by. By default, keys will be sorted in ascending order. To
            sort a key in descending order, prepend a ``"-"`` to the key name. For example, to sort
            the key value for ``"name"`` in descending order, use ``"-name"``.
        orders (list, optional): List of boolean sort orders to apply for each key. ``True``
            corresponds to ascending order while ``False`` is descending. Defaults to ``None``.
        reverse (bool, optional): Whether to reverse the sort. Defaults to ``False``.

    Returns:
        list: Sorted list.

    Example:

        >>> items = [{'a': 2, 'b': 1}, {'a': 3, 'b': 2}, {'a': 1, 'b': 3}]
        >>> results = order_by(items, ['b', 'a'])
        >>> assert results == [{'a': 2, 'b': 1},\
                               {'a': 3, 'b': 2},\
                               {'a': 1, 'b': 3}]
        >>> results = order_by(items, ['a', 'b'])
        >>> assert results == [{'a': 1, 'b': 3},\
                               {'a': 2, 'b': 1},\
                               {'a': 3, 'b': 2}]
        >>> results = order_by(items, ['-a', 'b'])
        >>> assert results == [{'a': 3, 'b': 2},\
                               {'a': 2, 'b': 1},\
                               {'a': 1, 'b': 3}]
        >>> results = order_by(items, ['a', 'b'], [False, True])
        >>> assert results == [{'a': 3, 'b': 2},\
                               {'a': 2, 'b': 1},\
                               {'a': 1, 'b': 3}]

    .. versionadded:: 3.0.0

    .. versionchanged:: 3.2.0
        Added `orders` argument.

    .. versionchanged:: 3.2.0
        Added :func:`sort_by_order` as alias.

    .. versionchanged:: 4.0.0
        Renamed from ``order_by`` to ``order_by`` and removed alias
        ``sort_by_order``.
    """
    if isinstance(collection, dict):
        collection = collection.values()

    # Maintain backwards compatibility.
    if pyd.is_boolean(orders):
        reverse = orders
        orders = None

    comparers = []

    if orders:
        for i, key in enumerate(keys):
            if pyd.has(orders, i):
                order = 1 if orders[i] else -1
            else:
                order = 1

            comparers.append((pyd.property_(key), order))
    else:
        for key in keys:
            if key.startswith("-"):
                order = -1
                key = key[1:]
            else:
                order = 1

            comparers.append((pyd.property_(key), order))

    def comparison(left, right):
        # pylint: disable=useless-else-on-loop,missing-docstring
        for func, mult in comparers:
            result = cmp(func(left), func(right))
            if result:
                return mult * result
        else:
            return 0

    return sorted(collection, key=cmp_to_key(comparison), reverse=reverse)


@overload
def partition(collection: Mapping[T, T2], predicate: DictIterateeT[T2, T, bool]) -> list[list[T2]]:
    ...


@overload
def partition(collection: Mapping[Any, T2], predicate: IterateeObjT = None) -> list[list[T2]]:
    ...


@overload
def partition(collection: Iterable[T], predicate: ListIterateeT[T, bool]) -> list[list[T]]:
    ...


@overload
def partition(collection: Iterable[T], predicate: IterateeObjT = None) -> list[list[T]]:
    ...


def partition(collection, predicate=None):
    """
    Creates an array of elements split into two groups, the first of which contains elements the
    `predicate` returns truthy for, while the second of which contains elements the `predicate`
    returns falsey for. The `predicate` is invoked with three arguments: ``(value, index|key,
    collection)``.

    If a property name is provided for `predicate` the created :func:`pluck` style predicate returns
    the property value of the given element.

    If an object is provided for `predicate` the created :func:`.matches` style predicate returns
    ``True`` for elements that have the properties of the given object, else ``False``.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        list: List of grouped elements.

    Example:

        >>> partition([1, 2, 3, 4], lambda x: x >= 3)
        [[3, 4], [1, 2]]

    .. versionadded:: 1.1.0
    """
    trues = []
    falses = []

    for is_true, value, _, _ in iteriteratee(collection, predicate):
        if is_true:
            trues.append(value)
        else:
            falses.append(value)

    return [trues, falses]


# TODO: check if can make this better
def pluck(collection: Iterable, path: str | list[str]) -> list:
    """
    Retrieves the value of a specified property from all elements in the collection.

    Args:
        collection (list): List of dicts.
        path (str|list): Collection's path to pluck

    Returns:
        list: Plucked list.

    Example:

        >>> pluck([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}], 'a')
        [1, 3, 5]
        >>> pluck([[[0, 1]], [[2, 3]], [[4, 5]]], '0.1')
        [1, 3, 5]
        >>> pluck([{'a': {'b': 1}}, {'a': {'b': 2}}], 'a.b')
        [1, 2]
        >>> pluck([{'a': {'b': [0, 1]}}, {'a': {'b': [2, 3]}}], 'a.b.1')
        [1, 3]
        >>> pluck([{'a': {'b': [0, 1]}}, {'a': {'b': [2, 3]}}], ['a', 'b', 1])
        [1, 3]

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Function removed.

    .. versionchanged:: 4.0.1
        Made property access deep.
    """
    return map_(collection, pyd.property_(path))


# TODO: gnagnagnagna
def reduce_(collection, iteratee=None, accumulator=None):
    """
    Reduces a collection to a value which is the accumulated result of running each element in the
    collection through the iteratee, where each successive iteratee execution consumes the return
    value of the previous execution.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed): Iteratee applied per iteration.
        accumulator (mixed, optional): Initial value of aggregator. Default is to use the result of
            the first iteration.

    Returns:
        mixed: Accumulator object containing results of reduction.

    Example:

        >>> reduce_([1, 2, 3, 4], lambda total, x: total * x)
        24

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed aliases ``foldl`` and ``inject``.
    """
    iterable = iterator(collection)

    if accumulator is None:
        try:
            _, accumulator = next(iterable)
        except StopIteration:
            raise TypeError("reduce_() of empty sequence with no initial value")

    result = accumulator

    if iteratee is None:
        iteratee = pyd.identity
        argcount = 1
    else:
        argcount = getargcount(iteratee, maxargs=3)

    for index, item in iterable:
        result = callit(iteratee, result, item, index, argcount=argcount)

    return result


# TODO: gnagnagnagna
def reduce_right(collection, iteratee=None, accumulator=None):
    """
    This method is like :func:`reduce_` except that it iterates over elements of a `collection` from
    right to left.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed): Iteratee applied per iteration.
        accumulator (mixed, optional): Initial value of aggregator. Default is to use the result of
            the first iteration.

    Returns:
        mixed: Accumulator object containing results of reduction.

    Example:

        >>> reduce_right([1, 2, 3, 4], lambda total, x: total ** x)
        4096

    .. versionadded:: 1.0.0

    .. versionchanged:: 3.2.1
        Fix bug where collection was not reversed correctly.

    .. versionchanged:: 4.0.0
        Removed alias ``foldr``.
    """
    if not isinstance(collection, dict):
        collection = list(collection)[::-1]

    return reduce_(collection, iteratee, accumulator)


# TODO: gnagnagnagna
def reductions(collection, iteratee=None, accumulator=None, from_right=False):
    """
    This function is like :func:`reduce_` except that it returns a list of every intermediate value
    in the reduction operation.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed): Iteratee applied per iteration.
        accumulator (mixed, optional): Initial value of aggregator. Default is to use the result of
            the first iteration.

    Returns:
        list: Results of each reduction operation.

    Example:

        >>> reductions([1, 2, 3, 4], lambda total, x: total * x)
        [2, 6, 24]

    Note:
        The last element of the returned list would be the result of using
        :func:`reduce_`.

    .. versionadded:: 2.0.0
    """
    if iteratee is None:
        iteratee = pyd.identity
        argcount = 1
    else:
        argcount = getargcount(iteratee, maxargs=3)

    results = []

    def interceptor(result, item, index):
        result = callit(iteratee, result, item, index, argcount=argcount)
        results.append(result)
        return result

    reducer = reduce_right if from_right else reduce_
    reducer(collection, interceptor, accumulator)

    return results


# TODO: gnagnagnagna
def reductions_right(collection, iteratee=None, accumulator=None):
    """
    This method is like :func:`reductions` except that it iterates over elements of a `collection`
    from right to left.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed): Iteratee applied per iteration.
        accumulator (mixed, optional): Initial value of aggregator. Default is to use the result of
            the first iteration.

    Returns:
        list: Results of each reduction operation.

    Example:

        >>> reductions_right([1, 2, 3, 4], lambda total, x: total ** x)
        [64, 4096, 4096]

    Note:
        The last element of the returned list would be the result of using
        :func:`reduce_`.

    .. versionadded:: 2.0.0
    """
    return reductions(collection, iteratee, accumulator, from_right=True)


@overload
def reject(
    collection: Mapping[T, T2], predicate: DictIterateeT[T2, T, bool] | IterateeObjT = None
) -> list[T2]:
    ...


@overload
def reject(
    collection: Iterable[T], predicate: ListIterateeT[T, bool] | IterateeObjT = None
) -> list[T]:
    ...


def reject(collection, predicate=None):
    """
    The opposite of :func:`filter_` this method returns the elements of a collection that the
    predicate does **not** return truthy for.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        list: Rejected elements of `collection`.

    Example:

        >>> reject([1, 2, 3, 4], lambda x: x >= 3)
        [1, 2]
        >>> reject([{'a': 0}, {'a': 1}, {'a': 2}], 'a')
        [{'a': 0}]
        >>> reject([{'a': 0}, {'a': 1}, {'a': 2}], {'a': 1})
        [{'a': 0}, {'a': 2}]

    .. versionadded:: 1.0.0
    """
    return [value for is_true, value, _, _ in iteriteratee(collection, predicate) if not is_true]


def sample(collection: Sequence[T]) -> T:
    """
    Retrieves a random element from a given `collection`.

    Args:
        collection (list|dict): Collection to iterate over.

    Returns:
        mixed: Random element from the given collection.

    Example:

        >>> items = [1, 2, 3, 4, 5]
        >>> results = sample(items)
        >>> assert results in items

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Moved multiple samples functionality to :func:`sample_size`. This
        function now only returns a single random sample.
    """
    return random.choice(collection)


def sample_size(collection: Sequence[T], n: int | None = None) -> list[T]:
    """
    Retrieves list of `n` random elements from a collection.

    Args:
        collection (list|dict): Collection to iterate over.
        n (int, optional): Number of random samples to return.

    Returns:
        list: List of `n` sampled collection values.

    Examples:

        >>> items = [1, 2, 3, 4, 5]
        >>> results = sample_size(items, 2)
        >>> assert len(results) == 2
        >>> assert set(items).intersection(results) == set(results)

    .. versionadded:: 4.0.0
    """
    num = min(n or 1, len(collection))
    return random.sample(collection, num)


@overload
def shuffle(collection: Mapping[Any, T]) -> list[T]:
    ...


@overload
def shuffle(collection: Iterable[T]) -> list[T]:
    ...


def shuffle(collection):
    """
    Creates a list of shuffled values, using a version of the Fisher-Yates shuffle.

    Args:
        collection (list|dict): Collection to iterate over.

    Returns:
        list: Shuffled list of values.

    Example:

        >>> items = [1, 2, 3, 4]
        >>> results = shuffle(items)
        >>> assert len(results) == len(items)
        >>> assert set(results) == set(items)

    .. versionadded:: 1.0.0
    """
    if isinstance(collection, dict):
        collection = collection.values()

    # Make copy of collection since random.shuffle works on list in-place.
    collection = list(collection)

    # NOTE: random.shuffle uses Fisher-Yates.
    random.shuffle(collection)

    return collection


def size(collection: Sized) -> int:
    """
    Gets the size of the `collection` by returning `len(collection)` for iterable objects.

    Args:
        collection (list|dict): Collection to iterate over.

    Returns:
        int: Collection length.

    Example:

        >>> size([1, 2, 3, 4])
        4

    .. versionadded:: 1.0.0
    """
    return len(collection)


def some(collection: Iterable[T], predicate: Callable[[T], Any] | None = None) -> bool:
    """
    # TODO: not true, not invoked with three arguments
    Checks if the predicate returns a truthy value for any element of a collection. The predicate is
    invoked with three arguments: ``(value, index|key, collection)``. If a property name is passed
    for predicate, the created :func:`map_` style predicate will return the property value of the
    given element. If an object is passed for predicate, the created :func:`.matches` style
    predicate will return ``True`` for elements that have the properties of the given object, else
    ``False``.

    Args:
        collection (list|dict): Collection to iterate over.
        predicate (mixed, optional): Predicate applied per iteration.

    Returns:
        bool: Whether any of the elements are truthy.

    Example:

        >>> some([False, True, 0])
        True
        >>> some([False, 0, None])
        False
        >>> some([1, 2, 3, 4], lambda x: x >= 3)
        True
        >>> some([1, 2, 3, 4], lambda x: x == 0)
        False

    .. versionadded:: 1.0.0

    .. versionchanged:: 4.0.0
        Removed alias ``any_``.
    """
    if predicate:
        cbk = pyd.iteratee(predicate)
        collection = (cbk(item) for item in collection)

    return any(collection)


@overload
def sort_by(
    collection: Mapping[Any, T2],
    iteratee: Callable[[T2], Any] | IterateeObjT = None,
    reverse: bool = False,
) -> list[T2]:
    ...


@overload
def sort_by(
    collection: Iterable[T],
    iteratee: Callable[[T], Any] | IterateeObjT = None,
    reverse: bool = False,
) -> list[T]:
    ...


def sort_by(collection, iteratee=None, reverse=False):
    """
    Creates a list of elements, sorted in ascending order by the results of running each element in
    a `collection` through the iteratee.

    Args:
        collection (list|dict): Collection to iterate over.
        iteratee (mixed, optional): Iteratee applied per iteration.
        reverse (bool, optional): Whether to reverse the sort. Defaults to ``False``.

    Returns:
        list: Sorted list.

    Example:

        >>> sort_by({'a': 2, 'b': 3, 'c': 1})
        [1, 2, 3]
        >>> sort_by({'a': 2, 'b': 3, 'c': 1}, reverse=True)
        [3, 2, 1]
        >>> sort_by([{'a': 2}, {'a': 3}, {'a': 1}], 'a')
        [{'a': 1}, {'a': 2}, {'a': 3}]

    .. versionadded:: 1.0.0
    """
    if isinstance(collection, dict):
        collection = collection.values()

    return sorted(collection, key=pyd.iteratee(iteratee), reverse=reverse)


#
# Utility methods not a part of the main API
#

# TODO: let's see if we need to type this better
def itermap(collection: CollectionT, iteratee: Callable | IterateeObjT = None) -> Generator:
    """Generative mapper."""
    for result in iteriteratee(collection, iteratee):
        yield result[0]
