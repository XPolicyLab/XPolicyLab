"""
Inspect, meta, etc.
"""

from __future__ import annotations

import functools
import inspect
import pprint
import sys
import types
from typing import Any, Dict, Literal
import warnings

from ..data_structure.tree_utils import is_mapping, is_sequence


def state_dict_class(keys: list[str]):
    """
    Just like pytorch nn.Module
    Add the following methods to the class:
        state_dict() -> dict of attribute keys
        load_state_dict(sdict)  restore states
    """

    def _wrap_class(cls):
        assert inspect.isclass(cls)

        def state_dict(self):
            return {k: getattr(self, k) for k in keys}

        def load_state_dict(self, states: Dict[str, Any]):
            if not set(keys).issubset(set(states.keys())):
                raise ValueError(f"states does not have all the required keys: {keys}")
            for k in keys:
                setattr(self, k, states[k])

        @property
        def state_keys(self):
            return keys

        cls.state_dict = state_dict
        cls.load_state_dict = load_state_dict
        cls.state_keys = state_keys
        return cls

    return _wrap_class


def implements_method(object, method: str):
    """
    Returns:
        True if object implements a method
    """
    return hasattr(object, method) and callable(getattr(object, method))


def assert_implements_method(object, method: str | list[str]):
    if isinstance(method, str):
        method = [method]
    for m in method:
        assert implements_method(object, m), (
            f"object {object.__class__} does not " f"implement method {m}()"
        )


def meta_decorator(decor):
    """
    a decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()

    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument.
      See stackoverflow: http://goo.gl/UEYbDB
    """
    import functools

    def single_callable(args, kwargs):
        return len(args) == 1 and len(kwargs) == 0 and callable(args[0])

    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_decorator
def make_recursive_func(fn, *, with_path=False):
    """
    Decorator that turns a function that works on a single array/tensor to working on
    arbitrary nested structures.
    """
    import functools

    import tree

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        if with_path:
            return tree.map_structure_with_path(
                lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct
            )
        else:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)

    return _wrapper


@meta_decorator
def deprecated(func, msg="", action="warning", type=""):
    """
    Function/class decorator: designate deprecation.

    Args:
      msg: string message.
      action: string mode
      - 'warning': (default) prints `msg` to stderr
      - 'noop': do nothing, just for source code annotation purposes
      - 'raise': raise DeprecatedError(`msg`)
    """
    action = action.lower()
    type = type.lower()
    ALL_ACTIONS = ["warn", "warning", "noop", "raise"]
    if action not in ALL_ACTIONS:
        raise ValueError(f"Unknown action {action}. Choose from {ALL_ACTIONS}.")
    ALL_TYPES = {
        "": DeprecationWarning,
        "pending": PendingDeprecationWarning,
        "future": FutureWarning,
    }
    if type not in ALL_TYPES:
        raise ValueError(f"Unknown type {type}. Choose from {ALL_TYPES.keys()}.")
    if not msg:
        msg = "This is a deprecated feature."

    WarningExceptionCls = ALL_TYPES[type]

    # only does the deprecation when being called
    @functools.wraps(func)
    def _deprecated(*args, **kwargs):
        if action in ["warning", "warn"]:
            warnings.warn(msg, WarningExceptionCls)
        elif action == "raise":
            raise WarningExceptionCls(msg)
        return func(*args, **kwargs)

    return _deprecated


@meta_decorator
def call_once(func, on_second_call: Literal["noop", "raise", "warn"] = "noop"):
    """
    Decorator to ensure that a function is only called once.

    Args:
      on_second_call (str): what happens when the function is called a second time.
    """
    assert on_second_call in [
        "noop",
        "raise",
        "warn",
    ], "mode must be one of 'noop', 'raise', 'warn'"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if wrapper._called:
            if on_second_call == "raise":
                raise RuntimeError(f"{func.__name__} has already been called. Can only call once.")
            elif on_second_call == "warn":
                warnings.warn(f"{func.__name__} has already been called. Should only call once.")
        else:
            wrapper._called = True
            return func(*args, **kwargs)

    wrapper._called = False
    return wrapper


class NoopObject:
    """
    Object that does nothing when called any method
    """

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    def __getattr__(self, name):
        def _func(*args, **kwargs):
            pass

        return _func


class NoopContext:
    """
    Placeholder context manager that does nothing.
    We could have written simply as:

    @contextmanager
    def noop_context(*args, **kwargs):
        yield

    but the returned context manager cannot be called twice, i.e.
    my_noop = NoopContext()
    with my_noop:
        do1()
    with my_noop: # trigger generator error
        do2()
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def make_registry_metaclass(class_name):
    """
    Usage:

      TrainerRegistry = make_registry_metaclass('TrainerRegistry')

      class BaseTrainer(metaclass=TrainerRegistry):
          pass

      class MyTrainer(BaseTrainer):
          pass

      TrainerRegistry['MyTrainer'] -> MyTrainer class  # syntax enabled by metaclass
      TrainerRegistry.get_class('MyTrainer')  # same as above
      TrainerRegistry.registry -> full dict of {name: trainer_class}

    Templated definition:
        class TrainerRegistry(type):
            registry = {}

            def __new__(cls, name, bases, attr):
                new_cls = super().__new__(cls, name, bases, attr)
                TrainerRegistry.registry[name] = new_cls
                return new_cls

            def get_class(cls, name):
                if name not in cls.registry:
                    raise KeyError(
                        f"Trainer class {name} not found in registry: "
                        f"{pprint.pformat(cls.registry)}"
                    )
                return cls.registry[name]"""

    def new__(cls, name, bases, attr):
        """
        Change the attr dict to dynamically add methods and attributes
        """
        new_cls = type.__new__(cls, name, bases, attr)
        cls.registry[name] = new_cls
        return new_cls

    def get_class(cls, name):
        if name not in cls.registry:
            existing_cls = list(cls.registry.keys())
            raise KeyError(f"{class_name} class '{name}' not found in registry: {existing_cls}")
        return cls.registry[name]

    def instantiate(cls_, cls, **kwargs):
        Cls = cls_.get_class(cls)
        return Cls(**kwargs)

    class _BracketOperator(type):
        def __getitem__(cls, name):
            return get_class(cls, name)

    return types.new_class(
        class_name,
        bases=(type,),
        kwds={"metaclass": _BracketOperator},
        exec_body=lambda ns: ns.update(
            {
                "registry": {},
                "__new__": new__,
                "get_class": classmethod(get_class),
                "instantiate": classmethod(instantiate),
            }
        ),
    )


class ClassRegistry:
    """
    May be a preferred way over make_registry_metaclass if your code does not support
    metaclass well, e.g. pickle or Ray

    Use in conjunction with `__init_subclass__` hook in your base class

    class BaseClass:
        registry = ClassRegistry()

        def __init_subclass__(cls, **kwargs):
            cls.registry.add(cls)
            super().__init_subclass__(**kwargs)

    print(registry)
    """

    def __init__(self, base_class_name: str = None):
        self.registry = {}
        self._base_class_name = base_class_name

    def add(self, cls):
        self.registry[cls.__name__] = cls

    def get(self, name):
        if name not in self.registry:
            existing_cls = list(self.registry.keys())
            base_name = self._base_class_name + " " if self._base_class_name else ""
            raise KeyError(f"{base_name} subclass '{name}' not found in registry: {existing_cls}")
        return self.registry[name]

    def __str__(self):
        return pprint.pformat(self.registry)

    def __getitem__(self, name):
        return self.get(name)

    def instantiate(self, cls, **kwargs):
        return self.get(cls)(**kwargs)


# ========================================================
# =================== Inspect utils ====================
# ========================================================


def func_parameters(func):
    return inspect.signature(func).parameters


def func_has_arg(func, arg_name):
    return arg_name in func_parameters(func)


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args


def enable_list_arg(func):
    """
    Function decorator.
    If a function only accepts varargs (*args),
    make it support a single list arg as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = pack_varargs(args)
        return func(*args, **kwargs)

    return wrapper


def enable_varargs(func):
    """
    Function decorator.
    If a function only accepts a list arg, make it support varargs as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = pack_varargs(args)
        return func(args, **kwargs)

    return wrapper


def pack_kwargs(args, kwargs):
    """
    Pack **kwargs or a single dict arg as dict

    def f(*args, **kwargs):
        kwdict = pack_kwargs(args, kwargs)
        # kwdict is now packed as a dict
    """
    if len(args) == 1 and is_mapping(args[0]):
        assert not kwargs, "cannot have both **kwargs and a dict arg"
        return args[0]  # single-dict
    else:
        assert not args, "cannot have positional args if **kwargs exist"
        return kwargs


def merge_kwargs(args, kwargs) -> Dict:
    """
    Merge all dicts in `args` and keywords in kwargs.

    E.g. merge_kwargs({"a.b": 1, "a.c": 2}, foo=6, bar=8)
       -> {"a.b": 1, "a.c": 2, "foo": 6, "bar": 8}
    """
    kw_all = {}
    for arg in args:
        assert is_mapping(arg), f"{arg} is not a dict."
        kw_all.update(arg)
    kw_all.update(kwargs)
    return kw_all


def enable_dict_arg(func):
    """
    Function decorator.
    If a function only accepts varargs (*args),
    make it support a single list arg as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = pack_kwargs(args, kwargs)
        return func(**kwargs)

    return wrapper


def enable_kwargs(func):
    """
    Function decorator.
    If a function only accepts a dict arg, make it support kwargs as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = pack_kwargs(args, kwargs)
        return func(kwargs)

    return wrapper


def has_keys(D, keys: list):
    assert is_mapping(D)
    return all(key in D for key in keys)


def assert_has_keys(D, keys: list):
    assert is_mapping(D), "Input is not a dict"
    for key in keys:
        if key not in D:
            raise KeyError(f'Required key "{key}" is missing in dict {D}')
    return True


def method_decorator(decorator):
    """
    Decorator of decorator: transform a decorator that only works on normal
    functions to a decorator that works on class methods
    From Django form: https://goo.gl/XLjxKK
    """

    @functools.wraps(decorator)
    def wrapped_decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            def bound_func(*args2, **kwargs2):
                return method(self, *args2, **kwargs2)

            return decorator(bound_func)(*args, **kwargs)

        return wrapper

    return wrapped_decorator


def accepts_varargs(func):
    """
    If a function accepts *args
    """
    params = inspect.signature(func).parameters
    return any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())


def accepts_kwargs(func):
    """
    If a function accepts **kwargs
    """
    params = inspect.signature(func).parameters
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def is_signature_compatible(func, *args, **kwargs):
    sig = inspect.signature(func)
    try:
        sig.bind(*args, **kwargs)
        return True
    except TypeError:
        return False


def make_list(x):
    """
    Turns a singleton object to a list. If already a list, no change.
    """
    if is_sequence(x):
        return x
    else:
        return [x]


def make_tuple(elem, repeats):
    """
    E.g. expand a singleton x into (x, x, x)
        useful for things like image_size or kernal, which can be a single int/float
        or a tuple of fixed size
    """
    if is_sequence(elem):
        assert len(elem) == repeats, f"length of input must be {repeats}: {elem}"
        return elem
    else:
        return (elem,) * repeats


def accumulate(iterable, fn=lambda x, y: x + y):
    """
    Return running totals
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


class DecoratorContextManager:
    """
    Allow a context manager to be used as a decorator
    From torch.auto_grad.grad_mode
    """

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)

        return decorate_context

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""

        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)

            # Generators are suspended and unsuspended at `yield`, hence we
            # make sure the grad mode is properly set every time the execution
            # flow returns into the wrapped generator and restored when it
            # returns through our `yield` to our caller (see PR #49017).
            cls = type(self)
            try:
                # Issuing `None` to a generator fires it up
                with cls():
                    response = gen.send(None)

                while True:
                    try:
                        # Forward the response to our caller and get its next request
                        request = yield response

                    except GeneratorExit:
                        # Inform the still active generator about its imminent closure
                        with cls():
                            gen.close()
                        raise

                    except BaseException:
                        # Propagate the exception thrown at us by the caller
                        with cls():
                            response = gen.throw(*sys.exc_info())

                    else:
                        # Pass the last request to the generator and get its response
                        with cls():
                            response = gen.send(request)

            # We let the exceptions raised above by the generator's `.throw` or
            # `.send` methods bubble up to our caller, except for StopIteration
            except StopIteration as e:
                # The generator informed us that it is done: take whatever its
                # returned value (if any) was and indicate that we're done too
                # by returning it (see docs for python's return-statement).
                return e.value

        return generator_context

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError
