from collections.abc import Collection

_INCOMPATIBLE_AUDIO: dict[str, frozenset[str]] = {
    '.mp4': frozenset({'vorbis'}),
    '.mov': frozenset({'opus'}),
    '.avi': frozenset({'vorbis', 'flac'}),
}


def needs_audio_reencode(
    audio_codec: str | None,
    output_suffix: str,
    supported_codecs: Collection[str],
) -> bool:
    if audio_codec is None:
        return False
    blocked = _INCOMPATIBLE_AUDIO.get(output_suffix.lower(), frozenset())
    return audio_codec.lower() in blocked or audio_codec not in supported_codecs
