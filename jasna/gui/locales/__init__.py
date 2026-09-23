"""Localization system for Jasna GUI."""

import json
import locale as _locale
import logging

from jasna.gui.paths import get_settings_path

from jasna.gui.locales.en import EN
from jasna.gui.locales.zh import ZH
from jasna.gui.locales.ja import JA
from jasna.gui.locales.ko import KO
from jasna.gui.locales.th import TH

logger = logging.getLogger(__name__)


TRANSLATIONS = {
    "en": EN,
    "zh": ZH,
    "ja": JA,
    "ko": KO,
    "th": TH,
}


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "简体中文",
    "ja": "日本語",
    "ko": "한국어",
    "th": "ไทย",
}


class LocaleManager:
    """Manages language selection and translation lookup."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_lang = "en"
        self._load()

    def _load(self):
        """Load language preference from settings."""
        path = get_settings_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._current_lang = data.get("language", "en")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read the language setting from %s: %s", path, e)
        else:
            # If settings.json is missing, try to autodetect system language
            try:
                lang, _ = _locale.getdefaultlocale()
                if lang and lang.startswith("zh"):
                    self._current_lang = "zh"
                elif lang and lang.startswith("ja"):
                    self._current_lang = "ja"
                elif lang and lang.startswith("ko"):
                    self._current_lang = "ko"
                elif lang and lang.startswith("th"):
                    self._current_lang = "th"
            except Exception:
                logger.debug("System locale autodetection failed; using 'en'", exc_info=True)

    def _save(self):
        """Save language preference to settings."""
        path = get_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read settings from %s, rewriting it: %s", path, e)

        data["language"] = self._current_lang

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning("Could not save the language setting to %s: %s", path, e)

    @property
    def current_language(self) -> str:
        return self._current_lang

    @property
    def available_languages(self) -> list[str]:
        return list(LANGUAGE_NAMES.keys())

    def set_language(self, lang: str):
        """Set and persist the current language."""
        if lang not in TRANSLATIONS:
            lang = "en"
        self._current_lang = lang
        self._save()

    def get(self, key: str, **kwargs) -> str:
        """Get translation for key. Falls back to English if not found."""
        translations = TRANSLATIONS.get(self._current_lang, TRANSLATIONS["en"])
        text = translations.get(key)

        # Fallback to English
        if text is None:
            text = TRANSLATIONS["en"].get(key, key)

        # Format with kwargs
        if kwargs:
            text = text.format(**kwargs)

        return text

    def __call__(self, key: str, **kwargs) -> str:
        """Shorthand for get()."""
        return self.get(key, **kwargs)


# Global instance
_locale = None


def get_locale() -> LocaleManager:
    """Get the global LocaleManager instance."""
    global _locale
    if _locale is None:
        _locale = LocaleManager()
    return _locale


def t(key: str, **kwargs) -> str:
    """Translate a key. Shorthand for get_locale().get(key)."""
    return get_locale().get(key, **kwargs)
