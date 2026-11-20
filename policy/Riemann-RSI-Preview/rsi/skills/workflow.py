import ast

TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, 'TryStar') else ())


def expression(value):
    return ast.parse(value, mode='eval').body if value is not None else None


def serialize(body):
    result = []
    for statement in body:
        if isinstance(statement, ast.If):
            node = {'kind': 'if', 'condition': ast.unparse(statement.test),
                    'then': serialize(statement.body), 'else': serialize(statement.orelse)}
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            node = {'kind': 'async_for' if isinstance(statement, ast.AsyncFor) else 'for',
                    'target': ast.unparse(statement.target), 'iterable': ast.unparse(statement.iter),
                    'body': serialize(statement.body), 'else': serialize(statement.orelse),
                    'type_comment': statement.type_comment}
        elif isinstance(statement, ast.While):
            node = {'kind': 'while', 'condition': ast.unparse(statement.test),
                    'body': serialize(statement.body), 'else': serialize(statement.orelse)}
        elif isinstance(statement, TRY_NODES):
            node = {'kind': 'try_star' if type(statement).__name__ == 'TryStar' else 'try',
                    'body': serialize(statement.body), 'handlers': [
                        {'exception': ast.unparse(handler.type) if handler.type is not None else None,
                         'name': handler.name, 'body': serialize(handler.body)} for handler in statement.handlers],
                    'else': serialize(statement.orelse), 'finally': serialize(statement.finalbody)}
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            node = {'kind': 'async_with' if isinstance(statement, ast.AsyncWith) else 'with',
                    'items': [{'context': ast.unparse(item.context_expr),
                               'target': ast.unparse(item.optional_vars) if item.optional_vars else None}
                              for item in statement.items],
                    'body': serialize(statement.body), 'type_comment': statement.type_comment}
        elif isinstance(statement, ast.Return):
            node = {'kind': 'return', 'value': ast.unparse(statement.value) if statement.value else None}
        elif isinstance(statement, ast.Break):
            node = {'kind': 'break'}
        elif isinstance(statement, ast.Continue):
            node = {'kind': 'continue'}
        else:
            node = {'kind': 'operation', 'code': ast.unparse(statement)}
        result.append(node)
    return result


def target(value):
    tree = ast.parse(value + ' = None').body[0]
    return tree.targets[0]


def compile_nodes(nodes):
    result = []
    if not isinstance(nodes, list):
        raise ValueError('Workflow nodes must be a declared list')
    for node in nodes:
        kind = node['kind']
        if kind == 'if':
            statement = ast.If(test=expression(node['condition']), body=compile_nodes(node['then']),
                               orelse=compile_nodes(node['else']))
        elif kind in ('for', 'async_for'):
            factory = ast.AsyncFor if kind == 'async_for' else ast.For
            statement = factory(target=target(node['target']), iter=expression(node['iterable']),
                body=compile_nodes(node['body']), orelse=compile_nodes(node['else']), type_comment=node.get('type_comment'))
        elif kind == 'while':
            statement = ast.While(test=expression(node['condition']), body=compile_nodes(node['body']),
                                  orelse=compile_nodes(node['else']))
        elif kind in ('try', 'try_star'):
            if kind == 'try_star' and not hasattr(ast, 'TryStar'):
                raise ValueError('Exception-group workflows require Python 3.11 or newer')
            factory = ast.TryStar if kind == 'try_star' else ast.Try
            statement = factory(body=compile_nodes(node['body']), handlers=[
                ast.ExceptHandler(type=expression(handler['exception']), name=handler['name'],
                                  body=compile_nodes(handler['body'])) for handler in node['handlers']],
                orelse=compile_nodes(node['else']), finalbody=compile_nodes(node['finally']))
        elif kind in ('with', 'async_with'):
            factory = ast.AsyncWith if kind == 'async_with' else ast.With
            statement = factory(items=[ast.withitem(context_expr=expression(item['context']),
                optional_vars=target(item['target']) if item['target'] is not None else None) for item in node['items']],
                body=compile_nodes(node['body']), type_comment=node.get('type_comment'))
        elif kind == 'return':
            statement = ast.Return(value=expression(node['value']))
        elif kind == 'break':
            statement = ast.Break()
        elif kind == 'continue':
            statement = ast.Continue()
        elif kind == 'operation':
            statements = ast.parse(node['code']).body
            if len(statements) != 1:
                raise ValueError('An operation node must contain exactly one statement')
            statement = statements[0]
        else:
            raise ValueError('Unknown workflow node: ' + kind)
        result.append(statement)
    return result


def count_nodes(nodes):
    result = len(nodes)
    for node in nodes:
        for key in ('body', 'then', 'else', 'finally'):
            if key in node:
                result += count_nodes(node[key])
        for handler in node.get('handlers', []):
            result += count_nodes(handler['body'])
    return result
