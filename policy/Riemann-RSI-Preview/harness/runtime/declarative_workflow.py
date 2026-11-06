"""Execute frozen skill graphs without evaluating Python code from JSON."""
from importlib import import_module

SCHEMA = "riemann-declarative-skill-workflow-v1"
LOOP_SCHEMA = "riemann-declarative-skill-workflow-v2"


class InvalidWorkflow(ValueError):
    pass


class _Returned(BaseException):
    def __init__(self, value):
        self.value = value


class _Break(BaseException):
    pass


class _Continue(BaseException):
    pass


def load_bindings(graph, package):
    """Resolve the graph's explicit local skill exports and exception classes."""
    def resolve(spec):
        module, symbol = spec.split(":")
        parts = module.split(".")
        if (len(parts) != 3 or parts[0] != "skills" or parts[-1] != "implementation"
                or any(not part.isidentifier() for part in parts)
                or not symbol.isidentifier() or symbol.startswith("_")):
            raise InvalidWorkflow("A binding requires an explicit local skill export")
        return getattr(import_module(package + "." + module), symbol)
    bindings = {name: resolve(spec) for name, spec in graph["bindings"].items()}
    errors = {name: resolve(spec) for name, spec in graph["errors"].items()}
    validate(graph, bindings, errors)
    return bindings, errors


def validate(graph, bindings, errors):
    required = {"schema", "mode", "official_action_horizon", "wam_reselection",
                "max_operations", "bindings", "errors", "steps"}
    extended = isinstance(graph, dict) and graph.get('schema') == LOOP_SCHEMA
    if (not isinstance(graph, dict) or set(graph) != required or graph["schema"] not in (SCHEMA, LOOP_SCHEMA)
            or graph["mode"] != "rule_only" or graph["wam_reselection"] is not False
            or type(graph["official_action_horizon"]) is not int
            or graph["official_action_horizon"] <= 0
            or type(graph["max_operations"]) is not int or not 1 <= graph["max_operations"] <= (65536 if extended else 4096)
            or set(graph["bindings"]) != set(bindings) or set(graph["errors"]) != set(errors)
            or any(not callable(value) for value in bindings.values())
            or any(not isinstance(value, type) or not issubclass(value, Exception) for value in errors.values())):
        raise InvalidWorkflow("Invalid frozen skill workflow contract")
    identifiers = set()
    count = 0
    def expression(value):
        if isinstance(value, dict):
            if set(value) == {"var"}:
                if not isinstance(value["var"], str) or not value["var"].isidentifier():
                    raise InvalidWorkflow("Invalid workflow result reference")
            else:
                for item in value.values():
                    expression(item)
        elif isinstance(value, list):
            for item in value:
                expression(item)
    def nodes(steps, depth=0, loops=0):
        nonlocal count
        if not isinstance(steps, list) or depth > 16:
            raise InvalidWorkflow("Workflow nesting must be bounded")
        for node in steps:
            count += 1
            if not isinstance(node, dict) or count > 1024:
                raise InvalidWorkflow("Invalid workflow node inventory")
            keys = set(node) - {"id"}
            if "id" in node:
                identifier = node["id"]
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise InvalidWorkflow("Workflow node identifiers must be unique")
                identifiers.add(identifier)
            if "call" in keys:
                if (keys - {"call", "args", "store"} or node["call"] not in bindings
                        or not isinstance(node.get("args", {}), dict)
                        or "store" in node and (not isinstance(node["store"], str) or not node["store"].isidentifier())):
                    raise InvalidWorkflow("Invalid declared skill invocation")
                expression(node.get("args", {}))
            elif "if" in keys:
                if keys != {"if", "then", "else"}:
                    raise InvalidWorkflow("Invalid workflow branch")
                expression(node["if"])
                nodes(node["then"], depth+1, loops)
                nodes(node["else"], depth+1, loops)
            elif "for_each" in keys:
                if (keys - ({'else'} if extended else set()) != {"for_each", "as", "max_items", "do"}
                        or not isinstance(node["as"], str) or not node["as"].isidentifier()
                        or type(node["max_items"]) is not int or not 1 <= node["max_items"] <= (4096 if extended else 64)):
                    raise InvalidWorkflow("Invalid bounded workflow iteration")
                expression(node["for_each"])
                nodes(node["do"], depth+1, loops+1)
                if 'else' in node:nodes(node['else'],depth+1,loops)
            elif extended and 'while' in keys:
                if (keys - {'else'} != {'while', 'args', 'max_iterations', 'do'}
                        or node['while'] not in bindings or not isinstance(node['args'],dict)
                        or type(node['max_iterations']) is not int
                        or not 1 <= node['max_iterations'] <= 4096):
                    raise InvalidWorkflow('Invalid bounded workflow while loop')
                expression(node['args'])
                nodes(node['do'],depth+1,loops+1)
                if 'else' in node:nodes(node['else'],depth+1,loops)
            elif extended and ('break' in keys or 'continue' in keys):
                if (loops == 0 or len(keys) != 1
                        or next(iter(keys)) not in ('break','continue')
                        or node[next(iter(keys))] is not True):
                    raise InvalidWorkflow('Loop control requires an enclosing loop')
            elif "try" in keys:
                if keys != {"try", "catch"} or not isinstance(node["catch"], list):
                    raise InvalidWorkflow("Invalid workflow error boundary")
                nodes(node["try"], depth+1, loops)
                for handler in node["catch"]:
                    if (not isinstance(handler, dict) or set(handler) != {"errors", "as", "do"}
                            or not isinstance(handler["errors"], list) or not handler["errors"]
                            or any(name not in errors for name in handler["errors"])
                            or not isinstance(handler["as"], str) or not handler["as"].isidentifier()):
                        raise InvalidWorkflow("Invalid declared error handler")
                    nodes(handler["do"], depth+1, loops)
            elif "raise" in keys:
                if keys != {"raise", "message"} or node["raise"] not in errors or not isinstance(node["message"], str):
                    raise InvalidWorkflow("Invalid declared workflow failure")
            elif "return" in keys:
                if keys != {"return"}:
                    raise InvalidWorkflow("Invalid workflow return")
                expression(node["return"])
            else:
                raise InvalidWorkflow("Workflow nodes must invoke skills or declare control flow")
    nodes(graph["steps"])


def execute(graph, context, bindings, errors):
    """Interpret declared calls, branches and bounded loops; skills own motion."""
    validate(graph, bindings, errors)
    values = {}
    operations = 0
    missing = object()
    def consume():
        nonlocal operations
        operations += 1
        if operations > graph['max_operations']:
            raise InvalidWorkflow('Declared workflow operation budget exhausted')
    def value(item):
        if isinstance(item, dict):
            if set(item) == {"var"}:
                try:
                    return values[item["var"]]
                except KeyError as error:
                    raise InvalidWorkflow("Workflow result used before assignment: " + item["var"]) from error
            return {key: value(child) for key, child in item.items()}
        if isinstance(item, list):
            return [value(child) for child in item]
        return item
    def scoped(name, item, body):
        previous = values.get(name, missing)
        values[name] = item
        try:
            run(body)
        finally:
            if previous is missing:
                del values[name]
            else:
                values[name] = previous
    def run(steps):
        for node in steps:
            consume()
            if "call" in node:
                result = bindings[node["call"]](context, **value(node.get("args", {})))
                if "store" in node:
                    values[node["store"]] = result
            elif "if" in node:
                run(node["then"] if value(node["if"]) else node["else"])
            elif "for_each" in node:
                items = value(node["for_each"])
                if not isinstance(items, (list, tuple)) or len(items) > node["max_items"]:
                    raise InvalidWorkflow("Workflow iteration exceeds its declared bound")
                broke=False
                for item in items:
                    try:
                        scoped(node["as"], item, node["do"])
                    except _Continue:
                        continue
                    except _Break:
                        broke=True
                        break
                if not broke:run(node.get('else',[]))
            elif 'while' in node:
                iterations=0
                broke=False
                while True:
                    consume()
                    if not bindings[node['while']](context,**value(node['args'])):
                        break
                    if iterations >= node['max_iterations']:
                        raise InvalidWorkflow('Declared workflow while iteration budget exhausted')
                    iterations += 1
                    try:
                        run(node['do'])
                    except _Continue:
                        continue
                    except _Break:
                        broke=True
                        break
                if not broke:run(node.get('else',[]))
            elif 'break' in node:
                raise _Break()
            elif 'continue' in node:
                raise _Continue()
            elif "try" in node:
                try:
                    run(node["try"])
                except Exception as error:
                    handler = next((item for item in node["catch"]
                        if isinstance(error, tuple(errors[name] for name in item["errors"]))), None)
                    if handler is None:
                        raise
                    scoped(handler["as"], error, handler["do"])
            elif "raise" in node:
                raise errors[node["raise"]](node["message"])
            elif "return" in node:
                raise _Returned(value(node["return"]))
    try:
        run(graph["steps"])
    except _Returned as result:
        return result.value
