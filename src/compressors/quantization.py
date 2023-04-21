import torch
import cornac


def get_post_dynamic(model: cornac.models.BiVAECF):
    model_quatized = torch.ao.quantization.quantize_dynamic(
        model.bivae,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    small_bivae = model.clone()
    small_bivae.bivae = model_quatized
    return small_bivae
