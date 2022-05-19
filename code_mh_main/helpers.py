
def removesuffix(path: str, suffix: str, /) -> str:
    if path.endswith(suffix):
        return path[:-len(suffix)]
    else:
        return path[:]

def removeprefix(self: str, prefix: str, /) -> str:
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]