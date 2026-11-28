import torch
from XPolicyLab.policy.vap.text_device import DeviceRoutedQwenPrompter

def test_routing_preserves_exact_provider_input_output_and_metadata():
    class Provider:
        tokenizer=object()
        manages_placement=True
        def encode_prompt(self,prompt,positive,device,**kw):
            self.received=(prompt,positive,str(device),kw)
            return torch.tensor([1.0,2.5,-3.0],dtype=torch.bfloat16,device=device)
    provider=Provider();routed=DeviceRoutedQwenPrompter(provider,'cpu')
    result=routed.encode_prompt('Exact unchanged prompt',positive=False,device='cpu',example=7)
    assert provider.received==('Exact unchanged prompt',False,'cpu',{'example':7})
    assert routed.tokenizer is provider.tokenizer and routed.manages_placement
    assert result.dtype==torch.bfloat16
    assert torch.equal(result,torch.tensor([1.0,2.5,-3.0],dtype=torch.bfloat16))
