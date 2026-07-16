from __future__ import annotations

from inspect import Parameter, signature


def callable_accepts_mode(func: object, parameter: str = "mode") -> bool:
    params = list(signature(func).parameters.values())
    if parameter != "mode":
        if any(param.kind == Parameter.VAR_KEYWORD for param in params):
            return True
        return any(param.name == parameter for param in params)

    if any(param.kind == Parameter.VAR_POSITIONAL for param in params):
        return True
    positional = {
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    }
    return len([param for param in params if param.kind in positional]) >= 3
