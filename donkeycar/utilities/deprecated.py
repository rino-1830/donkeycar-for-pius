"""非推奨機能をマークするデコレータを提供するモジュール。"""

import functools
import inspect
import warnings

string_types = (type(b''), type(u''))


def deprecated(reason):
    """関数やクラスを非推奨としてマークするデコレータ。

    このデコレータを付与した対象を使用すると警告が表示される。
    Laurent LaPorte 氏の素晴らしい回答
    https://stackoverflow.com/a/40301488/1733315 を参考にしている。
    """

    if isinstance(reason, string_types):

        # @deprecated が引数付きで使われた場合
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #        pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "非推奨のクラス {name} を呼び出しました ({reason})。"
            else:
                fmt1 = "非推奨の関数 {name} を呼び出しました ({reason})。"

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # @deprecated が引数なしで使われた場合
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #        pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "非推奨のクラス {name} を呼び出しました。"
        else:
            fmt2 = "非推奨の関数 {name} を呼び出しました。"

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
