"""Font-independent icons for GUI controls."""

import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk


def render_icon(name: str, size: int, color: str) -> Image.Image:
    scale = 4
    canvas_size = size * scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = canvas_size / 18

    def point(value: float) -> int:
        return round(value * unit)

    line_width = max(1, point(1.7))

    if name == "create":
        draw.line(
            [(point(4), point(9)), (point(14), point(9))],
            fill=color,
            width=line_width,
        )
        draw.line(
            [(point(9), point(4)), (point(9), point(14))],
            fill=color,
            width=line_width,
        )
    elif name == "delete":
        draw.line(
            [(point(3), point(5)), (point(15), point(5))],
            fill=color,
            width=line_width,
        )
        draw.line(
            [(point(7), point(3)), (point(11), point(3))],
            fill=color,
            width=line_width,
        )
        draw.rounded_rectangle(
            (point(5), point(6), point(13), point(16)),
            radius=point(1),
            outline=color,
            width=line_width,
        )
        draw.line(
            [(point(8), point(8)), (point(8), point(14))],
            fill=color,
            width=line_width,
        )
        draw.line(
            [(point(10), point(8)), (point(10), point(14))],
            fill=color,
            width=line_width,
        )
    elif name == "folder":
        draw.line(
            [
                (point(2), point(5)),
                (point(7), point(5)),
                (point(9), point(7)),
                (point(16), point(7)),
                (point(16), point(15)),
                (point(2), point(15)),
                (point(2), point(5)),
            ],
            fill=color,
            width=line_width,
            joint="curve",
        )
    elif name == "globe":
        bounds = (point(2), point(2), point(16), point(16))
        draw.ellipse(bounds, outline=color, width=line_width)
        draw.ellipse(
            (point(6), point(2), point(12), point(16)),
            outline=color,
            width=line_width,
        )
        draw.line(
            [(point(2), point(9)), (point(16), point(9))],
            fill=color,
            width=line_width,
        )
    elif name == "reset":
        draw.arc(
            (point(3), point(3), point(15), point(15)),
            start=35,
            end=330,
            fill=color,
            width=line_width,
        )
        draw.polygon(
            [
                (point(2), point(3)),
                (point(7), point(2)),
                (point(4), point(7)),
            ],
            fill=color,
        )
    elif name == "save":
        draw.rounded_rectangle(
            (point(3), point(2), point(15), point(16)),
            radius=point(1),
            outline=color,
            width=line_width,
        )
        draw.rectangle(
            (point(6), point(2), point(12), point(7)),
            outline=color,
            width=line_width,
        )
        draw.rectangle(
            (point(6), point(11), point(12), point(16)),
            outline=color,
            width=line_width,
        )
    else:
        raise ValueError(f"Unknown GUI icon: {name}")

    return image.resize((size, size), Image.Resampling.LANCZOS)


def create_icon(name: str, size: int, color: str) -> ctk.CTkImage:
    image = render_icon(name, size, color)
    return ctk.CTkImage(light_image=image, dark_image=image, size=(size, size))


def create_native_icon_image(master, name: str, size: int, color: str) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(render_icon(name, size, color), master=master)


class NativeIconButton(tk.Button):
    def __init__(
        self,
        master,
        name: str,
        icon_size: int,
        color: str,
        background: str,
        active_background: str,
        disabled_color: str,
        command: callable,
        width: int,
        height: int,
    ):
        self._enabled = True
        self._button_command = command
        self._icon_image = create_native_icon_image(master, name, icon_size, color)
        self._disabled_icon_image = create_native_icon_image(
            master, name, icon_size, disabled_color
        )
        super().__init__(
            master,
            image=self._icon_image,
            command=self._invoke,
            width=width,
            height=height,
            background=background,
            activebackground=active_background,
            relief="flat",
            overrelief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            cursor="hand2",
        )

    def _invoke(self) -> None:
        if self._enabled:
            self._button_command()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        state = kwargs.pop("state", None)
        if state is not None:
            self._enabled = state != "disabled"
            kwargs["image"] = (
                self._icon_image if self._enabled else self._disabled_icon_image
            )
        return super().configure(**kwargs)

    def cget(self, key: str):
        if key == "state":
            return "normal" if self._enabled else "disabled"
        return super().cget(key)


def render_toggle(
    selected: bool,
    width: int,
    height: int,
    track_color: str,
    button_color: str,
) -> Image.Image:
    scale = 4
    image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, width * scale - 1, height * scale - 1),
        radius=height * scale // 2,
        fill=track_color,
    )
    margin = 2 * scale
    diameter = (height - 4) * scale
    button_x = (width - height + 2) * scale if selected else margin
    draw.ellipse(
        (button_x, margin, button_x + diameter, margin + diameter),
        fill=button_color,
    )
    return image.resize((width, height), Image.Resampling.LANCZOS)
