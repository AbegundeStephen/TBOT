import os


def suffixed_path(path: str) -> str:
    """
    B11: when TBOT_INSTANCE is set (a second, side-by-side dry-run bot
    running a different PPL gear), insert its suffix into a log/state path
    so the two instances never write to the same file. A no-op when
    TBOT_INSTANCE is unset -- the live instance's paths are unchanged.

    "logs/trading_bot.log" + TBOT_INSTANCE=B -> "logs/trading_bot_B.log"
    "data/builder_state"   + TBOT_INSTANCE=B -> "data/builder_state_B"
    """
    inst = os.environ.get("TBOT_INSTANCE")
    if not inst:
        return path
    stripped = path.rstrip("/\\")
    trailing_slash = path[len(stripped):]
    root, ext = os.path.splitext(stripped)
    return f"{root}_{inst}{ext}{trailing_slash}"
