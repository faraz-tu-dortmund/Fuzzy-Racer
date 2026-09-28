"""A small Tkinter launcher for choosing a game mode."""

from __future__ import annotations

from collections.abc import Callable


def choose_mode() -> str | None:
    """Show the launcher and return the chosen action, or ``None`` if closed.

    The actions are ``"play"``, ``"ai"`` and ``"train"``.
    """
    import tkinter as tk

    from .assets import image_path
    from .settings import TITLE

    choice: list[str] = []
    root = tk.Tk()
    root.title(TITLE)
    root.geometry("400x400")
    root.resizable(False, False)
    root.iconphoto(True, tk.PhotoImage(file=str(image_path("icon_png.png"))))

    background = tk.PhotoImage(file=str(image_path("road.png")))
    canvas = tk.Canvas(root, width=400, height=400, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=background, anchor="nw")

    def pick(action: str) -> Callable[[], None]:
        def handler() -> None:
            choice.append(action)
            root.destroy()

        return handler

    buttons = (
        ("PLAY", "play", 0.3),
        ("LET NEURAL ENGINE PLAY", "ai", 0.5),
        ("TRAIN THE MODEL", "train", 0.7),
    )
    for text, action, rely in buttons:
        tk.Button(
            canvas,
            text=text,
            command=pick(action),
            bg="orange",
            activebackground="yellow",
            fg="white",
            width=30,
            height=3,
            font=("Arial", 12),
        ).place(relx=0.5, rely=rely, anchor="center")

    root.mainloop()
    return choice[0] if choice else None


def run_menu() -> None:
    """Keep showing the launcher and running the chosen mode until it is closed."""
    from .game import play, watch_ai
    from .training import train

    actions: dict[str, Callable[[], object]] = {
        "play": play,
        "ai": watch_ai,
        "train": train,
    }
    while (action := choose_mode()) is not None:
        result = actions[action]()
        print(result)
