"""Nonsecret desktop appearance preference, separate from node enrollment."""
import json
import os
from pathlib import Path

PALETTES = {
    'light': dict(background='#f3f6fb', card='#ffffff', text='#122743', muted='#65758c', button='#edf2fa', hover='#dbe7fa', track='#e8edf5', header='#122743'),
    'dark': dict(background='#101722', card='#1b2635', text='#edf3ff', muted='#a5b5ca', button='#29394d', hover='#364c68', track='#334258', header='#0c1420'),
}


def load_theme(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8')).get('theme')
        return value if value in PALETTES else 'light'
    except (OSError, ValueError, TypeError, AttributeError):
        return 'light'


def save_theme(path, theme):
    if theme not in PALETTES:
        raise ValueError('Unsupported theme')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.pending-' + str(os.getpid()))
    try:
        temporary.write_text(json.dumps({'theme': theme}) + '\n', encoding='utf-8')
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
