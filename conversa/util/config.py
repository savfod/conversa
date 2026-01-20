"""Configuration management for the conversational teacher."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from conversa.util.io import DEFAULT_SETTINGS_FILE, read_yaml, write_yaml

LEVEL_MAP = {
    "BEGINNER": "A1",
    "INTERMEDIATE": "B1",
    "ADVANCED": "C1",
}

# some may not work in speech-to-text / text-to-speech APIs
LANGUAGE_NAMES = {
    "aa": "Afar",
    "ab": "Abkhazian",
    "ae": "Avestan",
    "af": "Afrikaans",
    "ak": "Akan",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bi": "Bislama",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ce": "Chechen",
    "ch": "Chamorro",
    "co": "Corsican",
    "cr": "Cree",
    "cs": "Czech",
    "cu": "Church Slavic",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dv": "Divehi",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hr": "Croatian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "hz": "Herero",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ig": "Igbo",
    "ii": "Sichuan Yi",
    "ik": "Inupiaq",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kg": "Kongo",
    "ki": "Kikuyu",
    "kj": "Kuanyama",
    "kk": "Kazakh",
    "kl": "Kalaallisut",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kirghiz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lg": "Ganda",
    "li": "Limburgan",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lu": "Luba-Katanga",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mh": "Marshallese",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "na": "Nauru",
    "nb": "Bokmål, Norwegian",
    "nd": "Ndebele, North",
    "ne": "Nepali",
    "ng": "Ndonga",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "nr": "Ndebele, South",
    "nv": "Navajo",
    "ny": "Chichewa",
    "oc": "Occitan",
    "oj": "Ojibwa",
    "om": "Oromo",
    "or": "Oriya",
    "os": "Ossetian",
    "pa": "Panjabi",
    "pi": "Pali",
    "pl": "Polish",
    "ps": "Pushto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "rm": "Romansh",
    "rn": "Rundi",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sa": "Sanskrit",
    "sc": "Sardinian",
    "sd": "Sindhi",
    "se": "Northern Sami",
    "sg": "Sango",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "st": "Sotho, Southern",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tn": "Tswana",
    "to": "Tonga",
    "tr": "Turkish",
    "ts": "Tsonga",
    "tt": "Tatar",
    "tw": "Twi",
    "ty": "Tahitian",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Venda",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "za": "Zhuang",
    "zh": "Chinese",
    "zu": "Zulu",
}

ValidLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]

DEFAULT_CONFIG = {
    "target_language": "es",
    "teacher_language": "en",
    "level": "B1",
}


@dataclass
class Config:
    """Application configuration."""

    target_language: str
    teacher_language: str
    level: ValidLevel

    @classmethod
    def load(cls, settings_path: Path = DEFAULT_SETTINGS_FILE) -> "Config":
        """Load configuration from settings file.

        Creates default settings file if it doesn't exist.

        Args:
            settings_path: Path to settings file.

        Returns:
            Config instance.
        """
        if not settings_path.exists():
            write_yaml(DEFAULT_CONFIG, settings_path)
            print(f"Created settings file: \nfile://{settings_path}")
            print("Edit the file to customize your settings.")
            print()

        else:
            print(f"Loading settings from file:\nfile://{settings_path}")
            print()

        data = read_yaml(settings_path)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary.

        Args:
            data: Dictionary with config values.

        Returns:
            Config instance.
        """
        target_language = data.get("target_language", DEFAULT_CONFIG["target_language"])
        teacher_language = data.get("teacher_language", "")
        level = data.get("level", DEFAULT_CONFIG["level"])

        # Validate languages
        if target_language not in LANGUAGE_NAMES:
            raise ValueError(
                f"Unknown target_language: '{target_language}'. "
                f"Available: {list(LANGUAGE_NAMES.keys())}"
            )
        if teacher_language and teacher_language not in LANGUAGE_NAMES:
            raise ValueError(
                f"Unknown teacher_language: '{teacher_language}'. "
                f"Available: {list(LANGUAGE_NAMES.keys())}"
            )

        # Normalize level
        level_upper = str(level).upper()
        if level_upper in LEVEL_MAP:
            level_upper = LEVEL_MAP[level_upper]

        if level_upper not in ("A1", "A2", "B1", "B2", "C1", "C2"):
            print(f"Warning: Invalid level '{level}', using B1")
            level_upper = "B1"

        # Default teacher_language to target_language if empty
        if not teacher_language:
            teacher_language = target_language

        return cls(
            target_language=target_language,
            teacher_language=teacher_language,
            level=level_upper,  # type: ignore
        )

    @property
    def target_language_name(self) -> str:
        """Full name of target language (e.g., 'Spanish' for 'es')."""
        return LANGUAGE_NAMES[self.target_language]

    @property
    def teacher_language_name(self) -> str:
        """Full name of teacher language (e.g., 'English' for 'en')."""
        return LANGUAGE_NAMES[self.teacher_language]

    def print_settings(self) -> None:
        """Print current settings."""
        print(f"Target language: {self.target_language_name} ({self.target_language})")
        print(
            f"Teacher language: {self.teacher_language_name} ({self.teacher_language})"
        )
        print(f"Level: {self.level}")
