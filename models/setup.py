import ollama

print("Ollama 3.1:70b setup")
print("-----"*20)
ollama.pull("llama3.1:70b")
print("Ollama 3.1 setup")
print("-----"*20)
ollama.pull("llama3.1")
print("Mixtral 8x7b instruct setup")
print("-----"*20)
# ollama.pull("mixtral:8x7b")