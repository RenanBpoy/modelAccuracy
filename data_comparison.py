import sqlite3
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

def get_table_columns(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for table_name, in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema[table_name] = [col[1] for col in cursor.fetchall()]
    return schema

for i in range(count):
    example = dataset['train'][i]

    sql_context = example['sql_context']
    question = example['sql_prompt']
    sql_query = example['sql']

    # input para o modelo
    input_prompt = f'SQL Context: {sql_context} Question: {question}'
    inputs = tokenizer(input_prompt, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=100)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f'\nExemplo {i+1}')
    print(f'Prompt: {input_prompt}')
    print(f'Pergunta: {question}')
    print(f'SQL gerado: {generated_sql}')
    print(f'SQL esperado: {sql_query}')

    # tenta executar o sql context
    try:
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        for cur in sql_context.split(';'):
            cur = cur.strip()
            if cur:
                cursor.execute(cur)

        schema = get_table_columns(cursor)
        print(f'Esquema das tabelas: {schema}')

        # tenta executar sql esperado
        try:
            cursor.execute(sql_query)
            expected_result = cursor.fetchall()
            print(f'Resultado esperado: {expected_result}')

            # tenta executar sql gerado pelo modelo
            try:
                cursor.execute(generated_sql)
                generated_result = cursor.fetchall()
                print(f'Resultado gerado: {generated_result}')

                if expected_result == generated_result:
                    correct_count += 1
                else:
                    print(f'Diferença nos resultados: Esperado: {expected_result}, Gerado: {generated_result}')

            except sqlite3.Error as e:
                print(f'Erro ao executar SQL gerado: {e}')

        except sqlite3.Error as e:
            print(f'Erro ao executar SQL esperado: {e}')

    except sqlite3.Error as e:
        print(f'Erro ao criar banco SQLite: {e}')

    finally:
        conn.close()

accuracy = correct_count / count
print(f'\nPrecisão do modelo: {accuracy * 100:.2f}%')
