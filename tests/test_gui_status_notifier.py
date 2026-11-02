import pytest
from PIL import Image

jeepney = pytest.importorskip("jeepney")
low_level = pytest.importorskip("jeepney.low_level")
status_notifier = pytest.importorskip("jasna.gui.status_notifier")
StatusNotifierItem = status_notifier.StatusNotifierItem


def over_the_wire(message, serial: int):
    parser = low_level.Parser()
    parser.add_data(message.serialise(serial=serial))
    return parser.get_next_message()


def call(item, interface: str, member: str, signature=None, body=()):
    """Send one method call through the item and return its reply as the host would read it."""
    address = jeepney.DBusAddress("/StatusNotifierItem", "org.kde.StatusNotifierItem-1-1", interface)
    request = over_the_wire(jeepney.new_method_call(address, member, signature, body), serial=7)
    return over_the_wire(item.reply(request), serial=1)


def test_pixmap_is_argb_in_network_order():
    assert status_notifier.argb_pixmap(Image.new("RGBA", (1, 1), (1, 2, 3, 4))) == (1, 1, b"\x04\x01\x02\x03")


def test_host_reads_the_icon_and_the_current_title():
    item = StatusNotifierItem(Image.new("RGBA", (2, 2), (255, 0, 0, 255)), "Jasna", on_activate=lambda: None)
    item.title = "Jasna - 40% (1/3)"

    properties = call(item, "org.freedesktop.DBus.Properties", "GetAll", "s", ("org.kde.StatusNotifierItem",)).body[0]
    title = call(item, "org.freedesktop.DBus.Properties", "Get", "ss", ("org.kde.StatusNotifierItem", "Title")).body[0]

    assert properties["IconPixmap"] == ("a(iiay)", [(2, 2, b"\xff\xff\x00\x00" * 4)])
    assert properties["ToolTip"][1][2] == "Jasna - 40% (1/3)"
    assert title == ("s", "Jasna - 40% (1/3)")


@pytest.mark.parametrize("member", ["Activate", "SecondaryActivate", "ContextMenu"])
def test_any_click_opens_the_window(member):
    clicks = []
    item = StatusNotifierItem(Image.new("RGBA", (1, 1)), "Jasna", on_activate=lambda: clicks.append(member))

    reply = call(item, "org.kde.StatusNotifierItem", member, "ii", (10, 20))

    assert reply.header.message_type == jeepney.MessageType.method_return
    assert clicks == [member]


def test_unknown_calls_get_an_error():
    item = StatusNotifierItem(Image.new("RGBA", (1, 1)), "Jasna", on_activate=lambda: None)

    reply = call(item, "org.kde.StatusNotifierItem", "ProvideXdgActivationToken", "s", ("token",))

    assert reply.header.message_type == jeepney.MessageType.error
