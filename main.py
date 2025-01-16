from AIAssistantsLib.assistants import RAGAssistantGPT, RAGAssistantSber, RAGAssistantLocal, RAGAssistantYA, RAGAssistantMistralAI, RAGAssistantGGUF

with open('system_prompt_markdown.txt', 'r', encoding='utf-8') as f:
    sys_prompt = f.read()
assistant = RAGAssistantLocal(system_prompt=sys_prompt, kkb_path='data/vstore', model_name='/models/MiniCPM3-RAG-LoRA')

while True:
    user_input = 'Кто такие keyusers?'
    result = assistant.ask_question(user_input)
    print('Assistant:', result['answer'])