from collections.abc import Collection

import av

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


def aac_reencode_args(output_index: int) -> list[str]:
    return [f"-c:a:{output_index}", "aac", f"-b:a:{output_index}", "256k"]


def source_audio_codec(path: str) -> str | None:
    with av.open(path, metadata_errors="replace") as container:
        audio = container.streams.audio
        return audio[0].codec_context.codec.canonical_name if audio else None
