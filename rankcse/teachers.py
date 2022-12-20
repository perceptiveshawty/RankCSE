from .tool import *

# sentence-transformers/all-mpnet-base-v2
# voidism/diffcse-bert-base-uncased-sts
# princeton-nlp/sup-simcse-bert-base-uncased
# runs/my-sup-promcse-roberta-large

class Teacher(SimCSE):
    """
    A class for distilling ranking knowledge from SimCSE-based models. It is the same as the SimCSE except the features are precomputed and passed to the encode function.
    """

    def __init__(self, model_name_or_path: str = "voidism/diffcse-bert-base-uncased-sts", 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = "cls"):
        
        super().__init__(model_name_or_path, device, num_cells, num_cells_in_search, pooler)
        self.model = self.model.to(self.device if device is None else device)

    def encode(self, 
                inputs = None,
                device: str = "cuda:0", 
                return_numpy: bool = False,
                normalize_to_unit: bool = False,
                keepdim: bool = False,
                batch_size: int = 128,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        single_sentence = False

        embedding_list = [] 
        with torch.no_grad():
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
            if self.pooler == "cls":
                embeddings = outputs.pooler_output
            elif self.pooler == "cls_before_pooler":
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooler == "avg":
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state #First element of model_output contains all token embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-7)
                embeddings = sum_embeddings / sum_mask
                
            embedding_list.append(embeddings)
            embeddings = torch.cat(embedding_list)

        return embeddings
        