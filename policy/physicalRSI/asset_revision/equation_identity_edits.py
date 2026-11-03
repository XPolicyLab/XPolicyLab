"""Replace CAD-derived glyph templates/heights while preserving arithmetic logic."""
import ast


def edit_equation_reader(text):
    start=text.index("  t=np.load(Path(__file__).with_name('templates.npz'))")
    end=text.index('  self.glyphs=[]',start)
    text=text[:start]+'''  from physicalrsi_observed_geometry.generic_digits import silhouette_bank
  for digit,angle,mask in silhouette_bank():
   if -45<=angle<=45:
    self.templates.append((normalize(mask),str(digit),None,angle))
'''+text[end:]
    text=text.replace('from .geometry import TABLE_TOP, PAD_THICKNESS, OPERATOR_THICKNESS','from .geometry import TABLE_TOP, PAD_THICKNESS')
    text=text.replace('(normalize(m),label,OPERATOR_THICKNESS,0)','(normalize(m),label,None,0)')
    text=text.replace('def candidates(self,mask,operator,upright=False):','def candidates(self,mask,operator,upright=False,height=None):')
    text=text.replace('cost=cost,height=h,yaw_deg=angle','cost=cost,height=height,yaw_deg=angle')
    text=text.replace('def read(self,rgb,out,cached_number_masks=None,wrist_top=None,wrist_slot=None):',
        'def read(self,rgb,out,cached_number_masks=None,wrist_top=None,wrist_slot=None,*,observation=None):')
    text=text.replace("  out=Path(out);out.mkdir(parents=True,exist_ok=True)",'''  from physicalrsi_observed_geometry.equation_observation import EquationObservation
  from ..perception_services.perception.sam3_http import segment_many_http
  measured=EquationObservation(observation,segment_many_http)
  self.observed_plane=measured.plane
  out=Path(out);out.mkdir(parents=True,exist_ok=True)''',1)
    text=text.replace('  u,v=project(x,y,TABLE_TOP+PAD_THICKNESS)','  u,v=measured.rectification(x,y)')
    text=text.replace('    ranked=self.candidates(glyph,gi==2,slot is not None)',
        '    observed_height,height_evidence=measured.height(m,u,v)\n    ranked=self.candidates(glyph,gi==2,slot is not None,height=observed_height)')
    text=text.replace('perception_source=\'RGB_CAD_templates\'',"perception_source='generic_fonts_and_observed_RGB_height',height_evidence=height_evidence")
    text=text.replace('Symbol support comes from current RGB, CAD templates and visible structure.','Symbol support comes from current RGB, generic fonts and visible structure.')
    # The obsolete projection helper and imports are no longer needed here.
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project')
    lines[node.lineno-1:node.end_lineno]=[]
    text=''.join(lines).replace('from .geometry import TABLE_TOP, PAD_THICKNESS\n','').replace('F=640*10/22.212\n','')
    return text


def edit_equation_model(text):
    class BindObservation(ast.NodeTransformer):
        def visit_Dict(self,node):
            self.generic_visit(node)
            if len(node.keys)==1 and isinstance(node.keys[0],ast.Constant) and node.keys[0].value=='color':
                for key in ('intrinsic_matrix','extrinsic_matrix'):
                    node.keys.append(ast.Constant(key))
                    node.values.append(ast.parse(f"np.asarray(v[{key!r}])",mode='eval').body)
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='self.reader.read':
                node.keywords.append(ast.keyword(arg='observation',value=ast.parse('self.obs',mode='eval').body))
            return node
    return ast.unparse(ast.fix_missing_locations(BindObservation().visit(ast.parse(text))))+'\n'


def edit_equation_vision_gate(text):
    # Uncalled legacy photometric helper depended on fixed head calibration.
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='surface_evidence')
    lines[node.lineno-1:node.end_lineno]=[]
    return ''.join(lines).replace('from ..rgb_reader.reader import project\n','')
