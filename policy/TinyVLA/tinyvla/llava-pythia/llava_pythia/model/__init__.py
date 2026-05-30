from transformers import AutoConfig, AutoModelForCausalLM

from .language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from .language_model.pythia.llava_pythia import LlavaPythiaForCausalLM


try:
    AutoConfig.register(LlavaPythiaConfig.model_type, LlavaPythiaConfig)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(LlavaPythiaConfig, LlavaPythiaForCausalLM)
except ValueError:
    pass
