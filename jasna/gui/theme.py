"""Jasna GUI theme colors and styling constants."""


class Colors:
    # Backgrounds
    BG_MAIN = "#0f172a"      # Slate-900
    BG_PANEL = "#020617"     # Slate-950
    BG_CARD = "#1e293b"      # Slate-800
    
    # Borders
    BORDER = "#1e293b"       # Slate-800
    BORDER_LIGHT = "#334155" # Slate-700
    
    # Primary accent
    PRIMARY = "#4f46e5"      # Indigo-600
    PRIMARY_HOVER = "#4338ca"# Indigo-700
    PRIMARY_DARK = "#3730a3" # Indigo-800
    
    # Text
    TEXT_PRIMARY = "#cbd5e1"   # Slate-300
    
    # Status indicators
    STATUS_PROCESSING = "#34d399" # Emerald-400
    STATUS_ERROR = "#fb7185"      # Rose-400
    STATUS_PAUSED = "#fbbf24"     # Amber-400
    STATUS_COMPLETED = "#22c55e"  # Green-500
    STATUS_PENDING = "#94a3b8"    # Slate-400
    STATUS_CONFLICT = "#fbbf24"   # Amber-400 (output file exists)

    
    # Log colors
    LOG_INFO = "#34d399"    # Emerald-400
    LOG_WARNING = "#fbbf24" # Amber-400
    LOG_ERROR = "#fb7185"   # Rose-400
    LOG_DEBUG = "#94a3b8"   # Slate-400
    
    # Buy Me a Coffee branding
    BMC_YELLOW = "#FFDD00"        # Official Brand Color
    BMC_TEXT = "#0f172a"          # Slate-900 (dark text on yellow)


class Fonts:
    # Primary: Segoe UI (Windows system font, no embedding needed for PyInstaller)
    # Fallback chain handled by OS if unavailable
    FAMILY = "Segoe UI"
    # Monospace: Consolas (Windows system font)
    FAMILY_MONO = "Consolas"
    
    # Typography hierarchy per spec (+1px from original)
    SIZE_TITLE = 16      # App title "JASNA GUI" - bold
    SIZE_LARGE = 16      # Large UI elements (icons, +/- buttons)
    SIZE_BUTTON = 15     # Primary buttons - bold
    SIZE_NORMAL = 15     # Normal labels, input text
    SIZE_HEADING = 14    # Section headings - bold (small caps style)
    SIZE_SMALL = 13      # Secondary text, logs (mono)
    SIZE_TINY = 12       # Small caps labels "QUEUE", "SYSTEM OUTPUT" - bold


class Sizing:
    HEADER_HEIGHT = 48
    CONTROL_BAR_HEIGHT = 80
    LOG_PANEL_HEIGHT = 192
    QUEUE_PANEL_WIDTH = 320
    
    PADDING_LARGE = 16
    PADDING_MEDIUM = 12
    PADDING_SMALL = 8
    PADDING_TINY = 4
    
    BORDER_RADIUS = 6
    BUTTON_HEIGHT = 36
    INPUT_HEIGHT = 32
