"""Task-scoped manifests and paginated text; never execute uploaded code."""
import hashlib,json
from pathlib import Path

class SkillLibrary:
    def __init__(self,root,task_id):
        self.root=Path(root).resolve();self.task_id=task_id;self.entries={}
        if not task_id or Path(task_id).name!=task_id or task_id in ('.','..'):
            raise ValueError('Use an exact public task ID')
        folders=[self.root/'generic',self.root/'tasks'/task_id]
        for folder in folders:
            for path in sorted(folder.glob('*/manifest.json')):
                directory=path.parent.resolve()
                if not directory.is_relative_to(self.root) or path.is_symlink():raise ValueError('Skill path escapes frozen library')
                m=json.loads(path.read_text(encoding='utf-8'));key=m['id']
                if key in self.entries and folder.parent.name!='tasks':raise ValueError('Duplicate Skill ID: '+key)
                if m.get('kind') not in ('text','code'):raise ValueError('Service and learned policies are not part of strict L3')
                if m.get('interface')!='roborsi.public.v1':raise ValueError('Skill interface mismatch')
                text_path=directory/'SKILL.md'
                if text_path.is_symlink() or not text_path.resolve().is_relative_to(self.root):raise ValueError('Invalid Skill text path')
                text=text_path.read_text(encoding='utf-8')
                if m.get('task_context_file'):
                    doc_path=directory/m['task_context_file']
                    if not doc_path.resolve().is_relative_to(directory):raise ValueError('Task documentation escapes Skill directory')
                    docs=json.loads(doc_path.read_text(encoding='utf-8'))['tasks']
                    doc=docs.get(task_id)
                    if doc is not None:
                        text+='\n\n## Current task documentation\nTask: '+task_id+'\n\nInstruction:\n'+doc['instruction']+'\n\nDescription:\n'+doc['description']+'\n\nSource: '+doc['source']+'\n'
                    else:
                        text+='\nNo public documentation is registered for this task; follow the current episode public Instruction / Goal.\n'
                category='tool' if m.get('artifact_type') in ('tool','primitive') or m['kind']=='code' else 'skill'
                self.entries[key]={**m,'category':category,'scope':'task' if folder.parent.name=='tasks' else 'generic','text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}
        seen=set();active=set()
        def visit(key):
            if key in active:raise ValueError('Skill dependency cycle')
            if key not in self.entries:raise ValueError('Missing Skill dependency: '+key)
            if key in seen:return
            active.add(key)
            for dep in self.entries[key].get('dependencies',[]):visit(dep)
            active.remove(key);seen.add(key)
        for key in self.entries:visit(key)
    def catalog(self,category):
        return [{k:m[k] for k in ('id','description','kind','category','scope','version','stages','sha256')} for m in self.entries.values() if m['category']==category]
    def read(self,key,category,offset=0,stage=None):
        m=self.entries[key]
        if m['category']!=category:raise ValueError('Wrong catalog for '+key)
        if stage is not None and stage not in m['stages']:raise ValueError('Invalid Skill stage')
        text=m['text']
        if type(offset) is not int or not 0<=offset<=len(text):raise ValueError('Invalid offset')
        end=min(len(text),offset+6000)
        return {'id':key,'text':text[offset:end],'offset':offset,'next_offset':end if end<len(text) else None,'total_chars':len(text),'sha256':m['sha256']}
    def initial(self,budget=24000):
        entries=[m for m in self.entries.values() if m['category']=='skill' and m.get('load_mode')!='on_demand']
        entries.sort(key=lambda m:(m['scope']!='task',m['id']!='robot-action-timing',m['id']))
        result=[];used=0
        for m in entries:
            label='TASK Skill (strategy priority)' if m['scope']=='task' else 'GENERIC Skill (conditional fallback)'
            text=label+' '+m['id']+':\n'+m['text']
            # Text pagination remains available when initial context is full.
            if used+len(text)<=budget:result.append(text);used+=len(text)
        return 'Task Skill strategies take precedence over generic fallback guidance. Official interface and observation restrictions always prevail.\n\n'+'\n\n'.join(result)
