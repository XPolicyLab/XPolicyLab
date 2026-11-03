"""Conservative public prismatic-finger opening envelopes for placed-object checks."""
import numpy as np
import trimesh
from physicalrsi_observed_geometry.insert_obstacles import PlacedTubeScene

class FingerEnvelopeScene(PlacedTubeScene):
 def __init__(self,guard,bases,cylinders):
  super().__init__(guard,bases,cylinders)
  self.link_geometry={i:o.geometry for i,o in enumerate(guard.geometry.geometryObjects)}
  self.envelopes=[]
  for i,obj in enumerate(guard.geometry.geometryObjects):
   jid=obj.parentJoint
   if jid in {0,*guard.joint_ids}:continue
   joint=guard.model.joints[jid]
   if not joint.shortname().startswith('JointModelP') or joint.nq!=1:
    raise ValueError('Only public single-coordinate prismatic finger envelopes are supported')
   index=joint.idx_q;lower=guard.model.lowerPositionLimit[index];upper=guard.model.upperPositionLimit[index]
   if not np.isfinite([lower,upper]).all() or lower>upper:raise ValueError('Finite public finger travel required')
   vertices=np.asarray(trimesh.load(obj.meshPath,force='mesh').vertices)*obj.meshScale
   q=guard.pin.neutral(guard.model)
   guard.pin.updateGeometryPlacements(guard.model,guard.data,guard.geometry,guard.geometry_data,q)
   reference=guard.geometry_data.oMg[i].homogeneous.copy();points=[]
   for value in (lower,upper):
    q[index]=value;guard.pin.updateGeometryPlacements(guard.model,guard.data,guard.geometry,guard.geometry_data,q)
    relative=np.linalg.inv(reference)@guard.geometry_data.oMg[i].homogeneous
    points.append(vertices@relative[:3,:3].T+relative[:3,3])
   hull=trimesh.convex.convex_hull(np.vstack(points));mesh=self.fcl.BVHModelOBBRSS();mesh.beginModel(len(hull.faces),len(hull.vertices));vector_type=getattr(self.fcl,'StdVec_Vec3s',None) or self.fcl.StdVec_Vec3f;vv=vector_type();tt=self.fcl.StdVec_Triangle()
   for v in hull.vertices:vv.append(np.asarray(v,dtype=float))
   for a,b,c in hull.faces:tt.append(self.fcl.Triangle(int(a),int(b),int(c)))
   mesh.addSubModel(vv,tt);mesh.endModel()
   self.link_geometry[i]=mesh;self.links.append((i,float(np.linalg.norm(hull.vertices,axis=1).max())))
   self.envelopes.append({'link':obj.name,'joint':guard.model.names[jid],'lower':float(lower),'upper':float(upper),'hull_vertices':len(hull.vertices)})
 def clear(self,path,keys):
  for frame in path:
   for i,radius,world in frame:
    for key in keys:
     shape,obstacle,center,bound=self.obstacles[key]
     if np.linalg.norm(world[:3,3]-center)>radius+bound:continue
     result=self.fcl.CollisionResult();self.fcl.collide(self.link_geometry[i],self.fcl.Transform3f(world[:3,:3],world[:3,3]),shape,obstacle,self.fcl.CollisionRequest(),result)
     if result.isCollision():return False
  return True
