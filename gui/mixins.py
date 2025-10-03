import os


class LoggingMixin:
    def log(self, msg: str) -> None:
        print(f"[log] {self.__class__.__name__}: {msg}")


class ValidationMixin:
    def ensure_file_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
