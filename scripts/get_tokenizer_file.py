from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.save_vocabulary("/lus/eagle/projects/Local-LLM/shuyuanfan/Megatron-LM/scripts")
