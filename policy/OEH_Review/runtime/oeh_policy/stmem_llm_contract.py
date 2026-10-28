"""Declared bootstrap perception only; no memory-to-LLM review feedback loop."""
INITIAL_PURPOSES=frozenset({'initial_scene_discovery','initial_scene_confirmation','first_object_confirmation','first_entity_description','number_card_reading'})
REJECTED_PURPOSES=frozenset({'visual_review','periodic_discovery','legacy_periodic_description','reidentify'})


class MemoryLLMContractError(ValueError):pass


def authorize(interaction):
    if not isinstance(interaction,dict) or not isinstance(interaction.get('request_id'),str) or not interaction['request_id'].strip():
        raise MemoryLLMContractError('memory LLM calls require a named purpose and nonempty request_id')
    purpose=interaction.get('purpose')
    if isinstance(purpose,str) and purpose in REJECTED_PURPOSES:
        raise MemoryLLMContractError('Memory-to-LLM review feedback is disabled, including explicitly requested helper calls')
    if not isinstance(purpose,str) or purpose not in INITIAL_PURPOSES:raise MemoryLLMContractError('undeclared memory LLM purpose')
    return interaction


def ask(client,purpose,request_id,text,paths,deadline,*,explicit=False,**kwargs):
    interaction=authorize({'purpose':purpose,'request_id':request_id,'explicit':explicit})
    # Test doubles preserve their original transport interface; real clients enforce at the boundary too.
    if getattr(client,'requires_interaction',False):kwargs['interaction']=interaction
    return client.ask(text,paths,deadline,**kwargs)
