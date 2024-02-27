# NOTE: Do Following
# pip3 install accelerate
# python3 -m pip install -q -U git+https://github.com/huggingface/transformers.git
# use generate token from https://huggingface.co/settings/tokens and past it
# install python lib huggingface_hub
# pip install huggingface_hub
# python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOUR_TOKEN_HERE')"

from transformers import AutoTokenizer, AutoModelForCausalLM
def google_gemma():
    # instruction model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")
    # inference time
    # 1. 1.42 mins
    # 2. 1.16 mins
    while True:
        query = input("Enter your prompt: ")
        input_ids = tokenizer(query, return_tensors="pt").to(device)
        outputs = model.generate(**input_ids,max_new_tokens=300, do_sample=False)
        print("Query: ",query)
        print("Result")
        print(tokenizer.decode(outputs[0]))

google_gemma()
