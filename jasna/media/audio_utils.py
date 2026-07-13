_INCOMPATIBLE_AUDIO: dict[str, frozenset[str]] = {
    '.mp4': frozenset({'wmav1', 'wmav2', 'wmapro', 'vorbis'}),
    '.mov': frozenset({'wmav1', 'wmav2', 'wmapro', 'vorbis', 'opus'}),
    '.avi': frozenset({'opus', 'vorbis', 'flac'}),
    '.webm': frozenset({'aac', 'mp3', 'wmav1', 'wmav2', 'wmapro', 'dts', 'ac3', 'eac3', 'pcm_s16le', 'pcm_s24le', 'pcm_s32le', 'pcm_f32le'}),
}


def needs_audio_reencode(audio_codec: str | None, output_suffix: str) -> bool:
    if audio_codec is None:
        return False
    blocked = _INCOMPATIBLE_AUDIO.get(output_suffix.lower(), frozenset())
    return audio_codec.lower() in blocked
