from vllm import LLM, SamplingParams
llm = LLM(model="./Qwen3-1.7B-Q4_K_M.gguf", gpu_memory_utilization=0.6)

messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me a fun fact about giraffes."}]

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

outputs = llm.chat(messages, sampling_params)
for output in outputs:
    print(output.outputs[0].text)

from llama_cpp import Llama

llm = Llama(model_path="./Qwen3-1.7B-Q4_K_M.gguf", chat_format="Qwen3ForCasualLM")

messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me a fun fact about giraffes."}]

chat_response = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.7, stream=False)

print(chat_response['choices'][0]['message']['content'])
