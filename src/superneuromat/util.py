import os


def getenvbool(key, default=None):
    """Get the value of an environment variable or return a default."""
    s = os.environ.get(key, default)
    if isinstance(s, str):
        s2 = s.lower().strip()
        if s2 in ('false', '0'):
            return False
        elif s2 in ('true', '1'):
            return True
        if s == '':
            return default
    return s
