import torch

ug_sd = {}
sd = torch.load('FPT/pytorch_model.bin', map_location='cpu')

ug_sd['bert.joint_embeddings.map_fc.bias'] = sd['bert.embeddings.map_fc.bias']
ug_sd['bert.joint_embeddings.map_fc.weight'] = sd['bert.embeddings.map_fc.weight']
ug_sd['bert.joint_embeddings.LayerNorm.weight'] = sd['bert.embeddings.LayerNorm.weight']
ug_sd['bert.joint_embeddings.LayerNorm.bias'] = sd['bert.embeddings.LayerNorm.bias']

ug_sd['bert.multimodal_encoder.glyph_map.weight'] = sd['bert.embeddings.glyph_map.weight']
ug_sd['bert.multimodal_encoder.glyph_map.bias'] = sd['bert.embeddings.glyph_map.bias']
ug_sd['bert.multimodal_encoder.glyph_embeddings.embedding.weight'] = \
    sd['bert.embeddings.glyph_embeddings.embedding.weight']
ug_sd['bert.multimodal_encoder.pinyin_embeddings.conv.bias'] = \
    sd['bert.embeddings.pinyin_embeddings.conv.bias']
ug_sd['bert.multimodal_encoder.pinyin_embeddings.conv.weight'] = \
    sd['bert.embeddings.pinyin_embeddings.conv.weight']
ug_sd['bert.multimodal_encoder.pinyin_embeddings.embedding.weight'] = \
    sd['bert.embeddings.pinyin_embeddings.embedding.weight']
ug_sd['bert.multimodal_encoder.word_embeddings.weight'] = \
    sd['bert.embeddings.word_embeddings.weight']
ug_sd['bert.multimodal_encoder.position_embeddings.weight'] = \
    sd['bert.embeddings.position_embeddings.weight']
ug_sd['bert.multimodal_encoder.token_type_embeddings.weight'] = \
    sd['bert.embeddings.token_type_embeddings.weight']
ug_sd['bert.multimodal_encoder.position_ids'] = \
    sd['bert.embeddings.position_ids']

for key, value in sd.items():
    if key.startswith('bert.encoder'):
        ug_sd[key.replace('bert.encoder', 'bert.joint_corrector')] = value
    elif key.startswith('cls.predictions'):
        ug_sd[key.replace('cls.predictions', 'bert.predictions')] = value
    elif key.startswith('bert.embeddings'):
        continue
    else:
        ug_sd[key] = value

torch.save(ug_sd, 'UGFPT/pytorch_model.bin')

