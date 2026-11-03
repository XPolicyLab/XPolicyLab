"""Replace the language relay's preset plane with observed support geometry."""
import ast


def replace_language_relay_plane(text):
    tree=ast.parse(text)
    counts={'project':0,'unproject':0,'relay':0}
    class Bind(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name):
                if node.func.id in ('world_to_pixel_on_height','pixel_to_world_on_height'):
                    name='project' if node.func.id=='world_to_pixel_on_height' else 'unproject'
                    counts[name]+=1
                    return ast.Call(func=ast.Attribute(value=ast.Name(id='support',ctx=ast.Load()),attr=name,ctx=ast.Load()),args=node.args,keywords=[])
                if node.func.id=='free_relay_sites':
                    counts['relay']+=1
                    node.keywords=[ast.keyword(arg='support',value=ast.Name(id='support',ctx=ast.Load()))]
            return node
    tree=Bind().visit(tree)
    if counts!={'project':4,'unproject':1,'relay':2}:raise ValueError(f'Unexpected relay calls: {counts}')
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):
            if node.name=='free_relay_sites':
                if [arg.arg for arg in node.args.kwonlyargs]!=['camera']:raise ValueError('Expected bound relay camera')
                node.args.kwonlyargs[0].arg='support'
            elif node.name in ('plan_relay','relay_occupancy'):
                index=next(i for i,item in enumerate(node.body) if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='occupied' for t in item.targets))
                node.body[index:index]=ast.parse('support = observed_language_support(ep)').body
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    old="blocked = occupied.astype('uint8').copy()"
    if text.count(old)!=1:raise ValueError('Unexpected relay blocked mask')
    text=text.replace(old,old+'\n    blocked |= (~support.visible).astype(\'uint8\')')
    old='occupied = ep.inventory.last_robot.copy()'
    if text.count(old)!=2:raise ValueError('Unexpected relay occupancy masks')
    text=text.replace(old,old+' | ~support.visible')
    old='lower_z = z + 0.015'
    if text.count(old)!=1:raise ValueError('Unexpected relay release height')
    text=text.replace(old,'lower_z = z + support.height(sx, sy) - support.height(x, y) + 0.015')
    text='from physicalrsi_observed_geometry.language_support import observed_language_support\n'+text
    compile(text,'observed language relay plane','exec');return text
