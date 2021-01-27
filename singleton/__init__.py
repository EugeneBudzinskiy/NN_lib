from abc import ABCMeta


class SingletonMeta(ABCMeta):
    __instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instance:
            cls.__instance[cls] = super().__call__(*args, **kwargs)

        return cls.__instance[cls]
