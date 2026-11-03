"""Keep kong action logic while replacing fixed base and IK support geometry."""
import ast


def edit_kong_model(text):
    tree=ast.parse(text)
    episode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodeModel')
    constructor=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    for node in ast.walk(constructor):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='self.bases':node.value=ast.Dict(keys=[],values=[])
    reset=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
    reset.body[:0]=ast.parse('self.frames.reset_observations()\nself.bases = {}').body
    episode.body.append(ast.parse('''def update_obs(self,obs):
    super().update_obs(obs)
    self.bases = self.frames.bind_observation(obs)''').body[0])
    wrapper=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
    constructor=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    for node in ast.walk(constructor):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='self.ik':node.value=ast.Constant(None)
    class BindSolver(ast.NodeTransformer):
        def visit_Call(self,node):
            if ast.unparse(node.func)=='self.ik.solve_ik_to_joint':
                if len(node.args)!=3:raise ValueError('Unexpected kong IK call shape')
                node.func=ast.parse('self.frames.solve',mode='eval').body
                node.args=[ast.Name(id='a',ctx=ast.Load()),node.args[0],node.args[1]]
            return self.generic_visit(node)
    tree=BindSolver().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
