class ConwayException(Exception):
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self):
        return f"{super().__str__()} | Context: {self.context}"
