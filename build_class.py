import ast
import argparse
from pathlib import Path
from collections import defaultdict, deque

import pydash


WRAPPER_KW = "RES"
INIT_FILE = "src/pydash/__init__.py"
BASE_MODULE = """
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

{imports}

ValueT = TypeVar("ValueT", covariant=True)

{types}

class {class_name}:
"""



def build_header(class_name: str, imports: list[str]) -> str:
    import pydash.types
    types = Path(pydash.types.__file__).read_text()

    return BASE_MODULE.format(
        class_name=class_name, types=types, imports="\n".join(imports)
    )


def modules_and_api_funcs() -> dict[str, list[str]]:
    """This is mostly so we don't have to import `pydash`"""

    with open(INIT_FILE, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read())

    module_to_funcs = defaultdict(list)

    for node in ast.walk(tree):
        # TODO: maybe handle `Import` as well, not necessary for now
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                module_to_funcs[node.module].append(name.asname or name.name)

    return module_to_funcs


def is_overload(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "overload"
        for decorator in node.decorator_list
    )


def returns_typeguard(node: ast.FunctionDef) -> bool:
    def is_constant_typeguard(cst: ast.expr) -> bool:
        return isinstance(cst, ast.Constant) and "TypeGuard" in cst.value

    def is_subscript_typeguard(sub: ast.expr) -> bool:
        return (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and "TypeGuard" in sub.value.id
        )

    return (
        node.returns is not None
        and (
            is_constant_typeguard(node.returns)
            or is_subscript_typeguard(node.returns)
        )
    )


def has_single_default_arg(node: ast.FunctionDef) -> bool:
    return len(node.args.args) == 1 and len(node.args.defaults) >= 1


def chainwrapper_args(
    node: ast.FunctionDef,
) -> tuple[list[ast.expr], list[ast.keyword]]:
    # TODO: handle posonlyargs
    args: list[ast.expr] = [ast.Name(id=arg.arg) for arg in node.args.args[1:]]
    kwargs: list[ast.keyword] = [
        ast.keyword(arg=kw.arg, value=ast.Name(id=kw.arg))
        for kw in node.args.kwonlyargs
    ]

    if node.args.vararg:
        args.append(ast.Starred(value=ast.Name(id=node.args.vararg.arg)))

    if node.args.kwarg:
        kwargs.append(ast.keyword(value=ast.Name(id=node.args.kwarg.arg)))


    return args, kwargs


def wrap_type(wrapper: ast.Subscript, to_wrap: ast.expr) -> ast.expr:
    if isinstance(wrapper.slice, ast.Tuple):
        slice = ast.Tuple(
            elts=[
                s
                if not (isinstance(s, ast.Name) and s.id == WRAPPER_KW)
                else to_wrap
                for s in wrapper.slice.elts
            ]
        )
    else:
        slice = to_wrap

    return ast.Subscript(
        value=wrapper.value,
        slice=slice,
    )


def transform_function(node: ast.FunctionDef, wrapper: ast.Subscript) -> ast.FunctionDef:
    first_arg = node.args.args[0]
    cw_args, cw_kwargs = chainwrapper_args(node)

    if first_arg.annotation:
        first_arg.annotation = ast.Constant(
            value=ast.unparse(wrap_type(wrapper, first_arg.annotation))
        )

    first_arg.arg = "self"

    if node.returns:
        # TODO: `(some_arg: T) -> TypeGuard[T]` to `(some_arg: Any) -> bool`
        # TODO: otherwise we would get a `T` alone

        # change typeguard to bool as it is useless in a chain
        if returns_typeguard(node):
            node.returns = ast.Name(id="bool")

        node.returns = ast.Constant(
            value=ast.unparse(wrap_type(wrapper, node.returns))
        )

    if not is_overload(node):
        node.body = [
            ast.Return(
                value=ast.Call(
                    func=ast.Call(
                        func=ast.Name(id="self._wrap"),
                        args=[ast.Name(id=f"pyd.{node.name}")],
                        keywords=[],
                    ),
                    args=cw_args,
                    keywords=cw_kwargs,
                )
            )
        ]

    return node


def filename_from_module(module: str) -> str:
    return f"src/pydash/{module}.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("class_name")
    parser.add_argument("output", type=Path)
    parser.add_argument("wrapper")
    parser.add_argument("--imports", nargs="+")
    args = parser.parse_args()

    wrapper = ast.parse(args.wrapper).body[0]
    assert isinstance(wrapper, ast.Expr), "`wrapper` value should contain one expression"
    wrapper = wrapper.value
    assert isinstance(wrapper, ast.Subscript), "`wrapper` value should contain one with one subscript"

    to_file = open(args.output, "w")
    to_file.write(build_header(args.class_name, args.imports or []))

    module_to_funcs = modules_and_api_funcs()

    for module in module_to_funcs.keys():
        filename = filename_from_module(module)

        with open(filename, encoding="utf-8") as source:
            tree = ast.parse(source.read(), filename=filename)

        class_methods = deque()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_methods.extend(
                    f for f in node.body if isinstance(f, ast.FunctionDef)
                )

            # skipping class methods
            if node in class_methods:
                class_methods.popleft()
                continue

            if (
                isinstance(node, ast.FunctionDef)
                and node.name in module_to_funcs[module]
                and node.args.args  # skipping funcs without args for now
                and not has_single_default_arg(node)  # skipping 1 default arg funcs
            ):
                new_node = transform_function(node, wrapper)
                to_file.write(" " * 4)
                to_file.write(ast.unparse(new_node).replace("\n", f"\n{' ' * 4}"))
                to_file.write("\n\n")
                if new_node.name.endswith("_") and not is_overload(new_node):
                    to_file.write(
                        f"{' ' * 4}{new_node.name.rstrip('_')} = {new_node.name}"
                    )
                    to_file.write("\n\n")

    to_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
