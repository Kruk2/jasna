"""Linux tray icon over the StatusNotifierItem D-Bus protocol (KDE, Ubuntu, GNOME with AppIndicator).

X11 XEmbed tray icons are not used: on Wayland, KDE forwards clicks to them through input
emulation, which KWin may refuse, so a window hidden behind one could not come back.
"""

from __future__ import annotations

import logging
import os
import threading

from jeepney import DBusAddress, HeaderFields, MessageType, Properties, new_error, new_method_call, new_method_return, new_signal
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import open_dbus_connection
from PIL import Image

logger = logging.getLogger(__name__)

_ITEM_PATH = "/StatusNotifierItem"
_ITEM_INTERFACE = "org.kde.StatusNotifierItem"
_PROPERTIES_INTERFACE = "org.freedesktop.DBus.Properties"
_WATCHER = DBusAddress("/StatusNotifierWatcher", "org.kde.StatusNotifierWatcher", "org.kde.StatusNotifierWatcher")
_ACTIVATIONS = {"Activate", "SecondaryActivate", "ContextMenu"}
_POLL_SECONDS = 0.25
_REPLY_TIMEOUT_SECONDS = 2.0


def host_running() -> bool:
    """Whether a tray that shows StatusNotifierItems is running in this session."""
    try:
        with open_dbus_connection("SESSION") as connection:
            reply = connection.send_and_get_reply(
                Properties(_WATCHER).get("IsStatusNotifierHostRegistered"), timeout=_REPLY_TIMEOUT_SECONDS
            )
    except Exception as error:  # no session bus, auth refused, a hung bus...
        logger.debug("No system tray: %s", error)
        return False
    if reply.header.message_type == MessageType.error:  # no watcher
        return False
    return reply.body[0][1]


def argb_pixmap(image: Image.Image) -> tuple[int, int, bytes]:
    """The protocol's icon format: ARGB32 in network byte order."""
    red, green, blue, alpha = image.convert("RGBA").split()
    return image.width, image.height, Image.merge("RGBA", (alpha, red, green, blue)).tobytes()


class StatusNotifierItem:
    """Tray icon without a menu: any click calls ``on_activate`` on the D-Bus thread, and so
    does a tray that refuses the icon, so a hidden window never stays out of reach."""

    HAS_MENU = False

    def __init__(self, image: Image.Image, title: str, on_activate):
        self._pixmap = argb_pixmap(image)
        self._title = title
        self._on_activate = on_activate
        self._title_changed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value
        self._title_changed.set()

    def run_detached(self) -> None:
        connection = open_dbus_connection("SESSION")
        name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"
        try:
            connection.send_and_get_reply(message_bus.RequestName(name), timeout=_REPLY_TIMEOUT_SECONDS)
        except BaseException:
            connection.close()
            raise
        self._thread = threading.Thread(target=self._serve, args=(connection, name), name="tray-icon", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _serve(self, connection, name: str) -> None:
        try:
            self._serve_until_stopped(connection, name)
        except Exception:
            if self._stop.is_set():
                return
            logger.warning("The Jasna tray icon stopped working", exc_info=True)
            self._on_activate()

    def _serve_until_stopped(self, connection, name: str) -> None:
        with connection:
            # Not send_and_get_reply: it drops the host's property calls that arrive before the reply.
            connection.send(new_method_call(_WATCHER, "RegisterStatusNotifierItem", "s", (name,)))
            while not self._stop.is_set():
                if self._title_changed.is_set():
                    self._title_changed.clear()
                    emitter = DBusAddress(_ITEM_PATH, interface=_ITEM_INTERFACE)
                    connection.send(new_signal(emitter, "NewTitle"))
                    connection.send(new_signal(emitter, "NewToolTip"))
                try:
                    message = connection.receive(timeout=_POLL_SECONDS)
                except TimeoutError:
                    continue
                if message.header.message_type == MessageType.method_call:
                    connection.send(self.reply(message))
                elif message.header.message_type == MessageType.error:
                    logger.warning("The system tray refused the Jasna icon: %s", message.body)
                    self._on_activate()
                    return

    def properties(self) -> dict[str, tuple[str, object]]:
        return {
            "Category": ("s", "ApplicationStatus"),
            "Id": ("s", "jasna"),
            "Title": ("s", self._title),
            "Status": ("s", "Active"),
            "IconName": ("s", ""),
            "IconPixmap": ("a(iiay)", [self._pixmap]),
            "ToolTip": ("(sa(iiay)ss)", ("", [], self._title, "")),
            "ItemIsMenu": ("b", False),
            "Menu": ("o", "/NO_DBUSMENU"),
        }

    def reply(self, message):
        """The answer to one method call from the tray host."""
        interface = message.header.fields.get(HeaderFields.interface)
        member = message.header.fields.get(HeaderFields.member)
        if interface == _PROPERTIES_INTERFACE and member == "GetAll":
            return new_method_return(message, "a{sv}", (self.properties(),))
        if interface == _PROPERTIES_INTERFACE and member == "Get":
            value = self.properties().get(message.body[1])
            if value is not None:
                return new_method_return(message, "v", (value,))
        if interface in (_ITEM_INTERFACE, None) and member in _ACTIVATIONS:
            self._on_activate()
            return new_method_return(message)
        if interface in (_ITEM_INTERFACE, None) and member == "Scroll":
            return new_method_return(message)
        return new_error(message, "org.freedesktop.DBus.Error.UnknownMethod", "s", (f"Unknown method {member}",))
