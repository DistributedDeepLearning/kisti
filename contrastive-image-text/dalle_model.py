# Create a model from a vision encoder model and a text decoder model

from transformers import (
    VisionTextDualEncoderModel, 
    VisionTextDualEncoderProcessor, 
    AutoTokenizer, 
    AutoFeatureExtractor
)

import dalle_mini
from dalle_mini.data import Dataset
from dalle_mini.model import (
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer,
    set_partitions,
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
feat_ext = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(feat_ext, tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")