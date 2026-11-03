"""Replace kong's stored tile dimensions with current visible row estimates."""
import ast


def expr(text):return ast.parse(text,mode='eval').body


def strip(tree):
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'HEIGHT','THICK'} for t in n.targets))]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='vision':
            node.names=[name for name in node.names if name.name not in {'HEIGHT','THICK'}]
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and not n.names)]


class Dimensions(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id in {'HEIGHT','THICK'}:
            return expr("dimensions['%s']"%({'HEIGHT':'height_m','THICK':'thickness_m'}[node.id]))
        return node


def finish(tree):
    if any(isinstance(n,ast.Name) and n.id in {'HEIGHT','THICK'} for n in ast.walk(tree)):
        raise ValueError('Unconverted kong dimension reference')
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def edit_kong_dimension_vision(text):
    tree=ast.parse(text);strip(tree)
    tree.body[0]=ast.Expr(value=ast.Constant('RGB seam and ink matching with current camera, observed support and inferred row dimensions.'))
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        if node.name=='row_instances':
            index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='found')
            node.body[index:index]=ast.parse('''
from physicalrsi_observed_geometry import GeometryUnavailable
try:
    dimensions = measure_row_dimensions(rgb, (x,y,w,h), fold, seams, camera=camera, support_plane=support_plane)
except GeometryUnavailable as error:
    return [], {'bbox':[x,y,w,h], 'error':str(error)}
''').body
            for item in ast.walk(node):
                if isinstance(item,ast.Dict) and any(isinstance(k,ast.Constant) and k.value in {'xyz','seams'} for k in item.keys):
                    item.keys.append(ast.Constant('dimensions'));item.values.append(expr('dimensions'))
            Dimensions().visit(node)
        if node.name in {'flat_faces','declaration_stock'}:
            guard=next(n for n in node.body if isinstance(n,ast.If) and ast.unparse(n.test)=='row_info is None')
            guard.test=expr("row_info is None or 'dimensions' not in row_info")
            index=node.body.index(guard)+1
            node.body[index:index]=ast.parse("dimensions = row_info['dimensions']").body
            Dimensions().visit(node)
            if node.name=='declaration_stock':
                replaced=0
                for item in ast.walk(node):
                    if isinstance(item,ast.If) and '0.025 < short < 0.043' in ast.unparse(item.test):
                        item.test=expr("short <= 0 or long <= 0 or short > dimensions['width_m'] + 2 * dimensions['pixel_metric_m'] or long > dimensions['height_m'] + dimensions['height_uncertainty_m']")
                        replaced+=1
                    if isinstance(item,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='ring_green' for k in item.keys):
                        item.keys.append(ast.Constant('dimensions'));item.values.append(expr('dimensions'))
                if replaced!=1:raise ValueError('Expected stock dimension filter')
    tree.body+=ast.parse('''
def measure_row_dimensions(rgb, bbox, fold, seams, *, camera, support_plane):
    from physicalrsi_observed_geometry import GeometryUnavailable
    from physicalrsi_observed_geometry.upright_row import observe_upright_row
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    foreground = (white_mask(rgb)>0) | ((hsv[:,:,0]>30)&(hsv[:,:,0]<90)&(hsv[:,:,1]>65)&(hsv[:,:,2]>80))
    foreground = cv2.morphologyEx(foreground.astype('uint8'), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = bbox
    roi = np.zeros_like(foreground);roi[y:y+h,x:x+w]=1
    candidates=[]
    for contour in contours:
        mask=np.zeros_like(foreground);cv2.drawContours(mask,[contour],-1,1,-1)
        overlap=int(np.count_nonzero(mask&roi))
        if overlap:candidates.append((overlap,mask,cv2.boundingRect(contour)))
    if not candidates:raise GeometryUnavailable('Visible row outline required')
    _,mask,bounds=max(candidates,key=lambda item:item[0])
    return observe_upright_row(camera,mask,bounds,fold,seams,support_plane)
''').body
    return finish(tree)


def edit_kong_dimension_motion(text):
    tree=ast.parse(text);strip(tree)
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        if node.name in {'_front_push_plan','stock_pick_plan'}:
            source='tile' if node.name=='_front_push_plan' else 'stock'
            node.body[:0]=ast.parse(f"dimensions = {source}['dimensions']").body
        if node.name in {'handoff_plan','pickup_handoff_plan','upright_declaration_plan'}:
            node.args.kwonlyargs.append(ast.arg(arg='dimensions'));node.args.kw_defaults.append(None)
        Dimensions().visit(node)
    return finish(tree)


def edit_kong_dimension_model(text):
    tree=ast.parse(text);strip(tree)
    class Bind(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='THICK':return expr("tile['dimensions']['thickness_m']")
            return node
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id in {'handoff_plan','pickup_handoff_plan','upright_declaration_plan'}:
                node.keywords.append(ast.keyword(arg='dimensions',value=expr("self.stock['dimensions']")))
            return self.generic_visit(node)
    return finish(Bind().visit(tree))
