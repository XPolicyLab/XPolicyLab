"""Primary-model annotations are claims tied to saved evidence, never geometry measurements."""
import copy
import hashlib
import json


def update_schema():
    nullable = lambda n: {"type":["string","null"],"maxLength":n}
    return {"type":"object","additionalProperties":False,
        "properties":{"update_id":{"type":"string","minLength":1,"maxLength":120},
            "annotations":{"type":"array","maxItems":32,"items":{"type":"object","additionalProperties":False,
                "properties":{"entity_id":{"type":"string"},"observation_id":{"type":"string"},
                    "label":nullable(180),"description":nullable(800),
                    "completeness":{"type":["string","null"],"enum":["complete","partial","occluded",None]}},
                "required":["entity_id","observation_id","label","description","completeness"]}}},
        "required":["update_id","annotations"]}


def validate_update(value):
    if not isinstance(value,dict) or set(value)!={"update_id","annotations"}:
        raise ValueError("memory_update requires update_id and annotations")
    if not isinstance(value["update_id"],str) or not 1<=len(value["update_id"])<=120:
        raise ValueError("invalid update_id")
    annotations=value["annotations"]
    if not isinstance(annotations,list) or len(annotations)>32:raise ValueError("invalid annotations")
    for item in annotations:
        if not isinstance(item,dict) or set(item)!={"entity_id","observation_id","label","description","completeness"}:
            raise ValueError("invalid annotation fields")
        if any(not isinstance(item[k],str) or not item[k] for k in ("entity_id","observation_id")):
            raise ValueError("annotation requires entity and evidence IDs")
        for key,limit in (("label",180),("description",800)):
            if item[key] is not None and (not isinstance(item[key],str) or not item[key].strip() or len(item[key])>limit):
                raise ValueError("invalid annotation "+key)
        if item["completeness"] not in (None,"complete","partial","occluded"):
            raise ValueError("invalid completeness")


def apply_update(memory, value, now, call_id):
    validate_update(value)
    digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    cache=getattr(memory,"primary_updates",{})
    previous=cache.get(value["update_id"])
    if previous:
        if previous[0]!=digest:raise ValueError("update_id reused with different content")
        return copy.deepcopy(previous[1])
    items=[]
    for annotation in value["annotations"]:
        eid=annotation["entity_id"]; entity=memory.entities.get(eid)
        if entity is None:
            items.append({"entity_id":eid,"status":"unknown_entity"});continue
        evidence=[]
        # Only evidence actually archived under this identity can be referenced.
        for pattern in entity.get("patterns",[]):
            if pattern.get("observation_id")==annotation["observation_id"]:evidence.append(pattern)
        for observation in entity.get("tracking_observations",[])+entity.get("tracking_history",[])+memory.identity.tentative.get(eid,[]):
            if (observation.get("observation_id") or observation.get("evidence"))==annotation["observation_id"]:evidence.append(observation)
        state=entity.get("tracking_current") or entity["state"]
        if state.get("evidence")==annotation["observation_id"]:evidence.append(state)
        if not evidence:
            items.append({"entity_id":eid,"status":"unknown_evidence"});continue
        source={"kind":"primary_llm_annotation","call_id":call_id,"update_id":value["update_id"],
            "observation_id":annotation["observation_id"],"observed_at":evidence[0].get("time"),"known_at":now}
        if annotation["label"] is not None:
            entity.pop("context_label",None)
            entity["semantic_label"]=annotation["label"]
        if annotation["description"] is not None:
            entity["description_record"]={"text":annotation["description"],"source":source,
                "observation_id":annotation["observation_id"],"time":now}
            entity["description_status"]="primary_annotated"
        record={**copy.deepcopy(annotation),"source":source,"verified":False}
        entity["semantic_annotation"]=record
        entity.setdefault("semantic_annotation_history",[]).append(record)
        memory.log("primary_memory_update",record)
        memory.archive_summary(entity)
        items.append({"entity_id":eid,"status":"stored","verified":False})
    result={"update_id":value["update_id"],"items":items}
    cache[value["update_id"]]=(digest,copy.deepcopy(result));memory.primary_updates=cache
    return result


async def adjuncts(brain, envelope, packet, call_id):
    """Local pre-action operations; replies enter the next model context, not another helper call."""
    replies=getattr(brain,"memory_replies",[])
    worker=brain.runtime.stmem
    update=envelope.get("memory_update")
    queries=envelope.get("memory_query",[])
    if not isinstance(queries,list) or len(queries)>brain.runtime.cfg.get("stmem",{}).get("max_query_calls",4):
        replies.append(({"memory_query_response":{"status":"invalid_query_batch"}},[]));queries=[]
    if update is not None:
        try:result=await worker.update(update,packet["sim_time_s"],call_id)
        except (ValueError,TypeError,KeyError,RuntimeError,OSError,TimeoutError) as exc:
            result={"status":"update_failed","error":str(exc)[:300]}
        brain.runtime.emit("primary_memory_update",{"call_id":call_id,"result":result})
        replies.append(({"memory_update_response":result},[]))
    for query in queries:
        try:
            if not isinstance(query,dict) or set(query)!={"query_id","query"} or not isinstance(query["query_id"],str):
                raise ValueError("query requires query_id and query")
            is_task = isinstance(query["query"], dict) and query["query"].get("kind") == "task_tree_query"
            if is_task:
                from .task_tree_runtime import query as task_query
                result = task_query(brain.runtime, query["query"])
            else:
                result=await worker.query(query["query"],packet["sim_time_s"])
            images=result.pop("_image_items",[])
            result.pop("spatial_context",None)
            if not is_task and len(json.dumps(result))>24000:
                result={"status":"result_too_large","hint":"paginate with a smaller limit"};images=[]
        except (ValueError,TypeError,KeyError,RuntimeError,OSError,TimeoutError) as exc:
            result={"status":"query_failed","error":str(exc)[:300]};images=[]
        brain.runtime.emit("primary_memory_query",{"call_id":call_id,"query":query,"result":result})
        replies.append(({"memory_query_response":{"query_id":query.get("query_id") if isinstance(query,dict) else None,
            "result":result}},images))
    brain.memory_replies=replies
