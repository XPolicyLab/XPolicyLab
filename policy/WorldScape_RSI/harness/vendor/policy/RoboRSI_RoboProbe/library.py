"""Shared generic library; task-specific records stay outside the runtime."""
import hashlib,json
from pathlib import Path
class SkillLibrary:
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True); self.entries={}
        if any(p.is_dir() and p.name!='generic' for p in self.root.iterdir()): raise ValueError('Runtime contains a non-generic Skill directory')
        folder=self.root/'generic'
        if folder.is_symlink() or not folder.is_dir(): raise ValueError('Generic directory required')
        allowed={'id','version','interface','kind','stages','dependencies','artifact_type','description','implementation','load_mode'}
        for path in sorted(folder.glob('*/manifest.json')):
            directory=path.parent.resolve()
            if path.is_symlink() or path.parent.is_symlink() or not directory.is_relative_to(folder.resolve()): raise ValueError('Escaping Skill')
            m=json.loads(path.read_text()); key=m['id']
            if set(m)-allowed or key!=directory.name or key in self.entries: raise ValueError('Invalid generic manifest')
            if m.get('kind') not in ('text','code') or m.get('interface')!='roborsi.public.v1': raise ValueError('Invalid interface')
            if m.get('load_mode','on_demand') not in ('always','on_demand') or (m.get('load_mode')=='always' and m['kind']!='text'): raise ValueError('Invalid load mode')
            textpath=directory/'SKILL.md'
            if textpath.is_symlink() or not textpath.resolve().is_relative_to(directory): raise ValueError('Escaping text')
            text=textpath.read_text(); category='tool' if m.get('artifact_type') in ('tool','primitive') or m['kind']=='code' else 'skill'
            self.entries[key]={**m,'text':text,'category':category,'scope':'generic','sha256':hashlib.sha256(text.encode()).hexdigest()}
        seen,active=set(),set()
        def visit(k):
            if k in active or k not in self.entries: raise ValueError('Invalid Skill dependency')
            if k in seen:return
            active.add(k)
            for d in self.entries[k].get('dependencies',[]):visit(d)
            active.remove(k);seen.add(k)
        for k in self.entries:visit(k)
    def catalog(self,category):return [{k:m[k] for k in ('id','description','kind','category','scope','version','stages','sha256')} for m in self.entries.values() if m['category']==category]
    def read(self,key,category,offset=0,stage=None):
        m=self.entries[key]
        if m['category']!=category or (stage is not None and stage not in m['stages']):raise ValueError('Wrong category/stage')
        text=m['text']
        if type(offset) is not int or not 0<=offset<=len(text):raise ValueError('Invalid offset')
        end=min(len(text),offset+6000)
        return {'id':key,'text':text[offset:end],'offset':offset,'next_offset':end if end<len(text) else None,'total_chars':len(text),'sha256':m['sha256']}
    def initial(self,budget=24000):
        texts=['Shared Skill '+k+':\n'+m['text'] for k,m in sorted(self.entries.items()) if m.get('load_mode')=='always']
        result='\n\n'.join(texts)
        if len(result)>budget:raise ValueError('Preloaded generic contracts exceed budget; no silent omission allowed')
        return result
