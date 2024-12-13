from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

# carregando o dataset do Hugging Face https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
dataset = load_dataset('gretelai/synthetic_text_to_sql')

# carregando o modelo https://huggingface.co/cssupport/t5-small-awesome-text-to-sql
model_name = 'cssupport/t5-small-awesome-text-to-sql'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

correct_count = 0
count = 10

for i in range(count):
    example = dataset['train'][i]

    sql_context = example['sql_context']
    question = example['sql_prompt']
    sql_query = example['sql']

    # input do modelo contexto SQL (create table, inserts) + pergunta
    input_prompt = f'SQL Context: {sql_context} Question: {question}'

    inputs = tokenizer(input_prompt, return_tensors='pt', padding=True, truncation=True)

    # output do modelo
    outputs = model.generate(inputs['input_ids'], max_length=100)

    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print('Prompt:', input_prompt)
    print('Pergunta:', question)
    print('SQL gerado:', generated_sql)
    print('SQL esperado:', sql_query)
    print('\n')

    # verificar se o SQL gerado é igual ao esperado
    if (
        generated_sql.strip().lower().replace('"', "'").rstrip(';') ==
        sql_query.strip().lower().replace('"', "'").rstrip(';')
    ):
        correct_count += 1

accuracy = correct_count / count

print(f'Precisão do modelo: {accuracy * 100:.2f}%')
