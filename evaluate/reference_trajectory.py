"""Normalize legacy reference batches using the tools' declared input types."""

import ast
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple


class InvalidReferenceError(ValueError):
    """A legacy reference batch cannot be expanded without guessing."""


class ToolContract(NamedTuple):
    scalar_parameters: dict[str, tuple[type, ...]]
    required: tuple[str, ...]


def _scalar_types(annotation: ast.expr | None) -> tuple[type, ...] | None:
    """Return scalar types only when the annotation rules out list inputs."""
    if isinstance(annotation, ast.Constant):
        if annotation.value is None:
            return (type(None),)
        if isinstance(annotation.value, str):
            return _scalar_types(ast.parse(annotation.value, mode="eval").body)
    if isinstance(annotation, ast.Name):
        return {
            "str": (str,), "int": (int,), "float": (float,), "bool": (bool,),
        }.get(annotation.id)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        members = [_scalar_types(annotation.left), _scalar_types(annotation.right)]
    elif isinstance(annotation, ast.Subscript):
        name = getattr(annotation.value, "id", getattr(annotation.value, "attr", ""))
        if name == "Optional":
            members = [_scalar_types(annotation.slice), (type(None),)]
        elif name == "Union":
            nodes = (
                annotation.slice.elts
                if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            )
            members = [_scalar_types(node) for node in nodes]
        else:
            return None
    else:
        return None
    if any(member is None for member in members):
        return None
    return tuple(kind for member in members for kind in member)


def _read_tool_contracts(directory: Path) -> dict[str, ToolContract]:
    """Read MCP tool signatures without importing or executing tool servers."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Tool definitions directory does not exist: {directory}")
    contracts = {}
    for path in sorted(directory.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            exported = False
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if (
                    isinstance(target, ast.Attribute) and target.attr == "tool"
                    and isinstance(target.value, ast.Name) and target.value.id == "mcp"
                ):
                    exported = True
            if not exported:
                continue
            positional = node.args.posonlyargs + node.args.args
            required = [
                argument.arg for argument in positional[:len(positional) - len(node.args.defaults)]
            ]
            required.extend(
                argument.arg for argument, default
                in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None
            )
            scalar_parameters = {}
            for argument in positional + node.args.kwonlyargs:
                kinds = _scalar_types(argument.annotation)
                if kinds:
                    scalar_parameters[argument.arg] = kinds
            if node.name in contracts:
                raise ValueError(f"Ambiguous MCP tool name {node.name!r} in {directory}")
            contracts[node.name] = ToolContract(scalar_parameters, tuple(required))
    if not contracts:
        raise ValueError(f"No exported MCP tool definitions found in {directory}")
    return contracts


@lru_cache(maxsize=1)
def _tool_contracts() -> dict[str, ToolContract]:
    directory = Path(__file__).resolve().parents[1] / "agent" / "tools"
    return _read_tool_contracts(directory)


def normalize_reference_trajectory(reference: dict) -> dict:
    """Expand lists supplied to scalar parameters into repeated reference calls.

    Lists accepted by the declared interface (including scalar/list unions) and
    parameters without known scalar types are preserved. Fixed scalar inputs are
    shared by the repeated calls. Agent trajectories must not be normalized.
    """
    contracts = _tool_contracts()
    calls = []
    for step, call in enumerate(reference.get("tool_calls", []), 1):
        parameters = call.get("input", {})
        contract = contracts.get(call.get("name"))
        if contract is None or not isinstance(parameters, dict):
            calls.append(call)
            continue
        batches = {
            key: value for key, value in parameters.items()
            if key in contract.scalar_parameters and isinstance(value, list)
        }
        if not batches:
            calls.append(call)
            continue

        question = reference.get("question_index", "<unknown>")
        prefix = f"{question}: {call['name']} step {step}"
        missing = [key for key in contract.required if key not in parameters]
        if missing:
            raise InvalidReferenceError(f"{prefix} is missing required parameters: {', '.join(missing)}")
        lengths = [len(values) for values in batches.values()]
        if len(set(lengths)) != 1 or not lengths[0]:
            detail = ", ".join(f"{key}={len(values)}" for key, values in batches.items())
            raise InvalidReferenceError(
                f"{prefix} has empty or inconsistent batch lengths ({detail}); "
                "correct the reference before scoring"
            )
        for key, kinds in contract.scalar_parameters.items():
            if key not in parameters:
                continue
            values = batches[key] if key in batches else [parameters[key]]
            if not all(
                type(value) in kinds or (type(value) is int and float in kinds)
                for value in values
            ):
                raise InvalidReferenceError(f"{prefix} has invalid scalar values for {key}")

        for index in range(lengths[0]):
            calls.append({
                **call,
                "input": {
                    **parameters,
                    **{key: values[index] for key, values in batches.items()},
                },
            })
    return {**reference, "tool_calls": calls}
