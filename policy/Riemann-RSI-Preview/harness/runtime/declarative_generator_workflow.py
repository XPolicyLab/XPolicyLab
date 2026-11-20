"""Drive declared generator skills without changing their action/feedback cadence."""
from importlib import import_module
from inspect import isgenerator

from .declarative_workflow import InvalidWorkflow


SCHEMA = 'riemann-declarative-generator-workflow-v1'


class _Returned(BaseException):
    def __init__(self, value):
        self.value = value


def load_bindings(graph, package):
    def resolve(specification):
        if not isinstance(specification, str) or specification.count(':') != 1:
            raise InvalidWorkflow('A binding requires an explicit local skill export')
        module, symbol = specification.split(':')
        parts = module.split('.')
        if (len(parts) != 3 or parts[0] != 'skills' or parts[-1] != 'implementation'
                or any(not part.isidentifier() for part in parts)
                or not symbol.isidentifier() or symbol.startswith('_')):
            raise InvalidWorkflow('A binding requires an explicit local skill export')
        return getattr(import_module(package + '.' + module), symbol)
    bindings = {name: resolve(specification) for name, specification in graph['bindings'].items()}
    errors = {name: resolve(specification) for name, specification in graph['errors'].items()}
    validate(graph, bindings, errors)
    return bindings, errors


def validate(graph, bindings, errors):
    required = {'schema', 'mode', 'official_action_horizon', 'wam_reselection',
                'max_operations', 'bindings', 'errors', 'steps'}
    if (not isinstance(graph, dict) or set(graph) != required or graph['schema'] != SCHEMA
            or graph['mode'] != 'rule_only' or graph['wam_reselection'] is not False
            or type(graph['official_action_horizon']) is not int
            or not 1 <= graph['official_action_horizon'] <= 10000
            or type(graph['max_operations']) is not int
            or not 1 <= graph['max_operations'] <= 16384
            or not isinstance(graph['bindings'], dict) or not isinstance(graph['errors'], dict)
            or set(graph['bindings']) != set(bindings) or set(graph['errors']) != set(errors)
            or any(not callable(export) for export in bindings.values())
            or any(not isinstance(export, type) or not issubclass(export, Exception)
                   for export in errors.values())):
        raise InvalidWorkflow('Invalid frozen generator workflow contract')
    identifiers = set()
    node_count = 0

    def expression(item):
        if isinstance(item, dict):
            if set(item) == {'var'}:
                if not isinstance(item['var'], str) or not item['var'].isidentifier():
                    raise InvalidWorkflow('Invalid workflow result reference')
            else:
                for child in item.values():
                    expression(child)
        elif isinstance(item, list):
            for child in item:
                expression(child)

    def invocation(node):
        if (node['call'] not in bindings or not isinstance(node.get('args', {}), dict)
                or 'store' in node and (not isinstance(node['store'], str)
                                        or not node['store'].isidentifier())):
            raise InvalidWorkflow('Invalid declared generator skill invocation')
        expression(node.get('args', {}))

    def nodes(steps, depth=0):
        nonlocal node_count
        if not isinstance(steps, list) or depth > 16:
            raise InvalidWorkflow('Workflow nesting must be bounded')
        for node in steps:
            node_count += 1
            if not isinstance(node, dict) or node_count > 1024:
                raise InvalidWorkflow('Invalid workflow node inventory')
            keys = set(node) - {'id'}
            if 'id' in node:
                identifier = node['id']
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise InvalidWorkflow('Workflow node identifiers must be unique')
                identifiers.add(identifier)
            if 'call' in keys:
                if keys - {'call', 'args', 'store'}:
                    raise InvalidWorkflow('Invalid declared skill invocation')
                invocation(node)
            elif 'if' in keys:
                if keys != {'if', 'then', 'else'}:
                    raise InvalidWorkflow('Invalid workflow branch')
                expression(node['if'])
                nodes(node['then'], depth + 1)
                nodes(node['else'], depth + 1)
            elif 'for_each' in keys:
                if (keys != {'for_each', 'as', 'max_items', 'do'}
                        or not isinstance(node['as'], str) or not node['as'].isidentifier()
                        or type(node['max_items']) is not int or not 1 <= node['max_items'] <= 64):
                    raise InvalidWorkflow('Invalid bounded workflow iteration')
                expression(node['for_each'])
                nodes(node['do'], depth + 1)
            elif 'while' in keys:
                condition = node['while']
                if (keys != {'while', 'max_iterations', 'do'}
                        or not isinstance(condition, dict) or 'call' not in condition
                        or set(condition) - {'call', 'args'}
                        or type(node['max_iterations']) is not int
                        or not 1 <= node['max_iterations'] <= graph['official_action_horizon'] + 1):
                    raise InvalidWorkflow('Invalid bounded workflow retry')
                invocation(condition)
                nodes(node['do'], depth + 1)
            elif 'try' in keys:
                if keys != {'try', 'catch'} or not isinstance(node['catch'], list):
                    raise InvalidWorkflow('Invalid workflow error boundary')
                nodes(node['try'], depth + 1)
                for handler in node['catch']:
                    if (not isinstance(handler, dict) or set(handler) != {'errors', 'as', 'do'}
                            or not isinstance(handler['errors'], list) or not handler['errors']
                            or any(name not in errors for name in handler['errors'])
                            or not isinstance(handler['as'], str) or not handler['as'].isidentifier()):
                        raise InvalidWorkflow('Invalid declared error handler')
                    nodes(handler['do'], depth + 1)
            elif 'raise' in keys:
                if (keys != {'raise', 'message'} or node['raise'] not in errors
                        or not isinstance(node['message'], str)):
                    raise InvalidWorkflow('Invalid declared workflow failure')
            elif 'return' in keys:
                if keys != {'return'}:
                    raise InvalidWorkflow('Invalid workflow return')
                expression(node['return'])
            else:
                raise InvalidWorkflow('Workflow nodes must invoke skills or declare control flow')
    nodes(graph['steps'])


def execute(graph, context, bindings, errors):
    """Yield each original action; the caller supplies the next public observation."""
    validate(graph, bindings, errors)
    values = {}
    operations = 0
    action_count = 0
    missing = object()

    def operation():
        nonlocal operations
        operations += 1
        if operations > graph['max_operations']:
            raise InvalidWorkflow('Declared workflow operation budget exhausted')

    def value(item):
        if isinstance(item, dict):
            if set(item) == {'var'}:
                try:
                    return values[item['var']]
                except KeyError as error:
                    raise InvalidWorkflow('Workflow result used before assignment: ' + item['var']) from error
            return {key: value(child) for key, child in item.items()}
        if isinstance(item, list):
            return [value(child) for child in item]
        return item

    def drive(generator):
        nonlocal action_count
        try:
            while True:
                try:
                    action = next(generator)
                except StopIteration as result:
                    return result.value
                if not isinstance(action, dict) or not action:
                    raise InvalidWorkflow('Generator skills must yield nonempty action mappings')
                action_count += 1
                if action_count > graph['official_action_horizon']:
                    raise InvalidWorkflow('Declared native action horizon exhausted')
                yield action
        finally:
            generator.close()

    def scoped(name, item, body):
        previous = values.get(name, missing)
        values[name] = item
        try:
            yield from run(body)
        finally:
            if previous is missing:
                del values[name]
            else:
                values[name] = previous

    def run(steps):
        for node in steps:
            operation()
            if 'call' in node:
                result = bindings[node['call']](context, **value(node.get('args', {})))
                if isgenerator(result):
                    result = yield from drive(result)
                if 'store' in node:
                    values[node['store']] = result
            elif 'if' in node:
                yield from run(node['then'] if value(node['if']) else node['else'])
            elif 'for_each' in node:
                items = value(node['for_each'])
                if not isinstance(items, (list, tuple)) or len(items) > node['max_items']:
                    raise InvalidWorkflow('Workflow iteration exceeds its declared bound')
                for item in items:
                    yield from scoped(node['as'], item, node['do'])
            elif 'while' in node:
                condition = node['while']
                iteration = 0
                while True:
                    operation()
                    predicate = bindings[condition['call']](context, **value(condition.get('args', {})))
                    if isgenerator(predicate) or type(predicate) is not bool:
                        raise InvalidWorkflow('Retry predicates must return a bool without yielding actions')
                    if not predicate:
                        break
                    if iteration >= node['max_iterations']:
                        raise InvalidWorkflow('Declared workflow retry bound exhausted')
                    iteration += 1
                    yield from run(node['do'])
            elif 'try' in node:
                try:
                    yield from run(node['try'])
                except Exception as error:
                    handler = next((item for item in node['catch']
                                    if isinstance(error, tuple(errors[name] for name in item['errors']))), None)
                    if handler is None:
                        raise
                    yield from scoped(handler['as'], error, handler['do'])
            elif 'raise' in node:
                raise errors[node['raise']](node['message'])
            elif 'return' in node:
                raise _Returned(value(node['return']))
    try:
        yield from run(graph['steps'])
    except _Returned as result:
        return result.value
