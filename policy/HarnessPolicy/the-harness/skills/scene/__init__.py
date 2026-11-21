"""watch_scene, clear_view and press for programs, whichever grasping tools are on.

The three are implemented in skills/manipulation (their helpers are shared with its pick
and place), and the session class carries every skill's methods, so they run the same
whether the default tools by object id (skills/eef) or the manipulation skill's pick,
place and push are enabled. This skill only puts them in the program's namespace: they
used to come with the manipulation skill and were lost whenever eef replaced it.
"""


def exports(session):
    """The program's names of the three tools."""
    return {"watch_scene": session.watch_scene, "clear_view": session.clear_view, "press": session.press}


class SceneTools:
    """No methods of its own: see the module docstring."""
